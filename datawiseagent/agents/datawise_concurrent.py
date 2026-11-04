"""
DatawiseAgent automatically generates and executes Markdown and code cells to complete data-science tasks end-to-end. It follows an FST-based, multi-stage architecture with four implemented stages:
        1.	depth-first (DFS-style) Planning, 2) Incremental Execution, 3) Self-Debugging, and 4) Post-Filtering.

A single DatawiseAgent instance serves one user and manages all of that user’s sessions. A FastAPI service can host multiple agents to support multi-user, multi-session scenarios (registration, routing, and lifecycle management).

Key design points:
        •	The agent uses unified interaction representation and FST-based multi-stage archtecture (see the paper for details).
        •	Dual-format history. Chat history is stored both as plain messages and as notebook cells; the cell format is the agent’s inference substrate (via CellHistoryMemory).
        •	Pluggable code execution. Code can run in an isolated Docker sandbox or locally, implemented by DockerJupyterServer and LocalJupyterServer, respectively. For safety and reproducibility, Docker is strongly recommended.

Note: LocalJupyterServer has not been extensively tested. We strongly recommend using DockerJupyterServer.
"""

from __future__ import annotations

# TODO: load environment varibles more elegantly
from dotenv import load_dotenv

load_dotenv()
from enum import Enum
import uuid
from typing import List, Optional, Literal, Union, Tuple, TYPE_CHECKING, Dict
from pathlib import Path
from string import Template
from copy import deepcopy
from pathlib import Path
import os
from pydantic import BaseModel
from fastapi import UploadFile
import aiofiles

from datawiseagent.coding.jupyter import (
    DockerJupyterServer,
    LocalJupyterServer,
    JupyterCodeExecutor,
)
from datawiseagent.coding.code_utils import content_str
from datawiseagent.memory.session import Session, SessionContent
from datawiseagent.memory.chat_history import CellHistoryMemory
from datawiseagent.common.types import (
    LLMResult,
    FormatType,
    ConvertType,
    DatawiseAgentConfig,
    CodeExecutorConfig,
    UserInfo,
    SessionInfo,
)
from datawiseagent.common.types.cell import (
    NotebookCell,
    MarkdownCell,
    CodeCell,
    CodeOutputCell,
    UserCell,
    StepCell,
)
from datawiseagent.common.types.node import (
    Node,
    StepNode,
    ExecutionNode,
    DebugNode,
    PostDebuggingNode,
    UserNode,
    CompletionUsage,
)
from datawiseagent.common.log import logger
from datawiseagent.common.config import global_config
from datawiseagent.prompts.datawise import (
    # system
    DATAWISE_AGENT_SYSTEM_PROMPT,
    MULTI_STAGE_AGENT_WORKFLOW_PROMPT,
    CURRENT_WORKSPACE_STRUCTURE_AND_STATUS,
    # tag
    AWAIT_TAG,
    END_STEP_TAG,
    END_DEBUG_TAG,
    DEBUG_FAIL_TAG,
    DEBUG_SUCCEED_TAG,
    ITERATE_ON_LAST_STEP,
    ADVANCE_TO_NEXT_STEP,
    FULFILL_INSTRUCTION,
    # append code incrementally
    APPEND_CODE_WITH_PLANNING_PROMPT,
    APPEND_CODE_WITHOUT_PLANNING_PROMPT,
    # self debug iteratively
    DEBUGGING_APPEND_PROMPT,
    POST_DEBUGGING_APPEND_PROMPT,
    DEBUG_FAIL_TAG,
    DEBUG_SUCCEED_TAG,
    # planning
    INITIATE_WITH_PLANNING_PROMPT,
    PLANNING_APPEND_PROMPT,
    # append code with planning
    APPEND_CODE_WITH_PLANNING_PROMPT,
    CURRENT_STEP_GOAL_PROMPT,
    CURRENT_USER_INSTRCUTION_PROMPT,
    # work mode
    HYBRID_WORKFLOW,
)
from datawiseagent.prompts.datawise import *
from datawiseagent.agents.os.agent_pool_manager import agent_pool_manager
from datawiseagent.agents.mas import planner, code, debug, filter
from datawiseagent.agents.mas.debug_trace import (
    create_debug_trace_record,
    clone_debug_trace_record,
)
from datawiseagent.memory.candidate_runtime import CandidateSessionView
import datawiseagent.agents.utils.execution as execution
import datawiseagent.agents.utils.parser as parser
import datawiseagent.agents.utils.prompt_tool as prompt_tool
import time
import inspect
import importlib
import asyncio
import traceback
import random
from datawiseagent.agents.os.experience import experience_collecting
from datawiseagent.agents.os.network.calculate_network_latency import get_network_latency
from datawiseagent.agents.os.network.real_network_latency import get_real_network_latency
from datawiseagent.llms import load_llm
from typing import Any

if TYPE_CHECKING:
    from datawiseagent.management import DatawiseAgentManager

# For backward compatibility in function signatures (e.g., create_new_session),
# we define a module-level `llm_config` that points to the default configuration
# provided by the agent_pool_manager.
llm_config = agent_pool_manager.default_llm_config
agent_config = global_config["agent"]
code_executor_config = global_config["code_executor"]
USERS_LOG_PATH = global_config["log"]["users_log_path"]

SESSIONS_LOG_PATH = global_config["log"]["sessions_log_path"]

from datawiseagent.agents.os.physical_agent import PhysicalAgent
from datawiseagent.agents.os.process_management import bestone


class DatawiseMASConcurrent:
    """
    A core class representing the Datawise Agent.

    Key responsibilities
    --------------------
    - Session lifecycle
      Creates, tracks, persists, and tears down per-user sessions and their isolated workspaces.

    - Code generation & execution
      Produces Markdown/Code cells via an LLM, executes code cells in Jupyter kernels,
      and attaches structured outputs (stdout/stderr/exit code, rich display).

    - Planning workflow
      Optionally uses a plan/append/update loop with step goals to structure multi-step tasks.

    - Self-debugging
      Iteratively repairs failing cells with bounded turns, then re-verifies execution.

    - Persistence & broadcasting
      Saves session state to JSON; broadcasts updates to subscribers over WebSocket.

    Attributes
    ----------
    user_id : uuid.UUID
        Unique identifier of the user that owns this agent.
    user_name : Optional[str]
        Optional display name for the user.
    user_root_dir : pathlib.Path
        Root directory on disk for this user's persisted files.
    sessions : dict[uuid.UUID, Session]
        In-memory map from session ID to `Session` objects.
    llm_config : dict
        Effective LLM configuration used by this agent.
    llm : BaseChatModel
        Bound chat LLM instance, created from `llm_config`.
    agent_manager : Optional[DatawiseAgentManager]
        Manager used for WebSocket broadcasting.

    Notes
    -----
    - Per-user workspace: For each user, the agent creates a dedicated user workspace at `user_root_dir`. All sessions belonging to that user live under this directory, each in its own subdirectory that serves as the session’s working area.
    - Session persistence: For each session, the agent persists the session’s conversation/notebook history as JSON files under the directory derived from SESSIONS_LOG_PATH. In other words, all sessions for a given user are saved beneath this path, one JSON per session (plus updates).
    - Methods that run code are `async` and must be awaited.
    """

    _global_task_counter = 0 # Global counter for sequential task IDs


    def __init__(
        self,
        user_id: uuid.UUID,
        user_name: Optional[str] = None,
        agent_manager: Optional[DatawiseAgentManager] = None,
    ) -> None:
        """
        Initialize a `DatawiseMASConcurrent` bound to a single user's workspace.

        Parameters
        ----------
        user_id : uuid.UUID
            Owner user ID.
        user_name : Optional[str], default=None
            Optional display name.
        agent_manager : Optional[DatawiseAgentManager], default=None
            Manager for WebSocket broadcasting and user/agent registry.
        """
        self.user_id = user_id
        self.user_name = user_name

        self.user_root_dir = Path(USERS_LOG_PATH) / str(
            user_id
        )  # relative to working directory if USERS_LOG_PATH is a relative path.
        self.user_root_dir.mkdir(parents=True, exist_ok=True)

        self.sessions: dict[uuid.UUID, Session] = {}

        # Hold an instance of the global LLM manager.
        self.agent_llm_manager = agent_pool_manager

        # Populate LLM attributes from the LLM manager.
        self.llm: BaseChatModel = self.agent_llm_manager.default_llm
        self.llm_config: dict = self.agent_llm_manager.default_llm_config
        self.llms: dict[str, BaseChatModel] = self.agent_llm_manager.llms
        
        # Assign the WebSocket/user manager from the __init__ parameter.
        self.agent_manager = agent_manager
        
        self.last_small_step_agent = None
        self.last_small_step_agent_zone = "Local"
        self.real_mode = bool(global_config.get("agent", {}).get("real_mode", False))
        self.network_mode = global_config.get("agent", {}).get(
            "network_mode", "real_probe" if self.real_mode else "simulated"
        )
        _probe_cfg = global_config.get("agent", {}).get("real_probe", {}) or {}
        self.real_probe_samples = int(_probe_cfg.get("samples", 3))
        self.real_probe_timeout_sec = float(_probe_cfg.get("timeout_sec", 5.0))
        self.real_probe_cache_ttl_sec = float(_probe_cfg.get("cache_ttl_sec", 30.0))
        self.real_probe_fallback_latency_sec = float(_probe_cfg.get("fallback_latency_sec", 0.8))
        logger.info(
            f"Concurrent mode settings: real_mode={self.real_mode}, network_mode={self.network_mode}"
        )

    def _get_agent_config(self, role: str, agent_name: str) -> Dict:
        return self.agent_llm_manager.agent_configs.get(role, {}).get(agent_name, {})

    def _compute_network_latency(
        self,
        source_zone: Optional[str],
        target_zone: Optional[str],
        cells: Optional[list],
        *,
        role: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> tuple[float, int]:
        payload_cells = cells or []
        if self.real_mode and self.network_mode == "real_probe":
            target_agent_config = {}
            if role and agent_name:
                target_agent_config = self._get_agent_config(role, agent_name)
            latency, cell_size, _ = get_real_network_latency(
                source_zone,
                target_zone,
                payload_cells,
                target_agent_config=target_agent_config,
                probe_samples=self.real_probe_samples,
                probe_timeout_sec=self.real_probe_timeout_sec,
                cache_ttl_sec=self.real_probe_cache_ttl_sec,
                fallback_latency_sec=self.real_probe_fallback_latency_sec,
            )
            return latency, cell_size
        return get_network_latency(source_zone, target_zone, payload_cells)

    async def create_new_session(
        self,
        reset_llm_config: dict = llm_config,
        reset_code_executor: dict = code_executor_config,
        session_name: Optional[str] = None,
        tool_mode: Literal["default", "dsbench", "datamodeling"] = "default",
    ) -> uuid.UUID:
        """
        Create and register a new session, optionally overriding LLM and executor configs.

        The session workspace is initialized and optional few-shot cells are injected
        based on `tool_mode`. A Jupyter code executor (local/Docker) is provisioned.

        Parameters
        ----------
        reset_llm_config : dict, default=llm_config
            Per-session override for LLM configuration. If different from the current
            agent-level config, the agent's `llm` will be reloaded.
        reset_code_executor : dict, default=code_executor_config
            Serialized `CodeExecutorConfig` fields to construct a Jupyter executor.
        session_name : Optional[str], default=None
            Human-friendly session name for persistence and listing.
        tool_mode : {"default", "dsbench", "datamodeling"}, default="default"
            Controls the initial bootstrap cells and environment checks.

        Returns
        -------
        uuid.UUID
            The newly created session ID.

        Raises
        ------
        AssertionError
            If the code executor cannot be constructed.
        """

        # user could specify the llm config manually for each session.
        if reset_llm_config != llm_config:
            self.llm_config = reset_llm_config
            self.llm = load_llm(deepcopy(reset_llm_config))

        session = Session(
            chat_history=CellHistoryMemory(),
            user_root_dir=self.user_root_dir,
            user_id=self.user_id,
            session_name=session_name,
        )
        session.code_executor_config = CodeExecutorConfig(**reset_code_executor)

        if session.code_executor_config.use_docker:
            session.code_executor = await JupyterCodeExecutor.create(
                jupyter_server=DockerJupyterServer(
                    # custom_image_name="my-jupyter-image-gpus",
                    custom_image_name=session.code_executor_config.image_name,
                    out_dir=session.root_dir,
                    auto_remove=True,
                    stop_container=True,
                    use_proxy=session.code_executor_config.use_proxy,
                    use_gpu=session.code_executor_config.use_gpu,
                ),
                # all rich text data output by the kernel will be stored in the path of `{session.session_root_dir}/display`
                output_dir=session.display_dir,
                use_docker_space=True,
            )
        else:

            session.code_executor = await JupyterCodeExecutor.create(
                # Jupyter Server has added heartbeat mechanism to keep the connection between jupyter kernel client and jupyter kernel. We should consider that situation. When user/agent want to execute the code, kernel should be open (restart or check the connection), and websocket should be connected. At other time, the disconnection is tolerable, especially the kernel shutdown. Because it's much lower cost to check and keep the websocket connected, but it's high-cost to keep the kernel open.
                # 1. So if kernel shutdown while  heartbeat checking, let it go.
                # 2. If the case happens while agent is running the code, restart the kernel and rerun all.
                # 3. If the case happens while user is asking something, restart the kernel and rerun all.
                jupyter_server=LocalJupyterServer(out_dir=session.root_dir),
                # all rich text data output by the kernel will be stored in the path of `{session.session_root_dir}/display`
                output_dir=session.display_dir,
                use_docker_space=False,
            )
        session.agent_config = agent_config
        # register a new session in self.sessions
        self.sessions[session.session_id] = session

        await self.chat_history_initialization(session, tool_mode)

        return session.session_id

    async def chat_history_initialization(
        self,
        session: Session,
        tool_mode: Literal["default", "dsbench", "datamodeling"] = "default",
    ):
        """
        Bootstrap a new session with environment overview and verification cells.

        Depending on `tool_mode`, installs/prints package versions and optionally
        uploads internal support modules (e.g., `dsbench`) into the session's workspace.

        Parameters
        ----------
        session : Session
            Target session to initialize.
        tool_mode : {"default", "dsbench", "datamodeling"}, default="default"
            Which preset to apply.

        Notes
        -----
        - Executes the bootstrap cells immediately and stores their outputs.
        - When `tool_mode="dsbench"`, Python source files from the `dsbench` package are
          serialized and uploaded into the session's `system` directory.
        """

        if tool_mode == "datamodeling":
            m1 = MarkdownCell(
                content="# Development Environment Overview\nThe development environment is managed using `Conda` with `Python 3.10` as the interpreter. Some basic libraries for data analysis and machine learning have already been installed to streamline common tasks.",
                role="assistant",
                name="Datawise_Agent",
            )

            m2 = MarkdownCell(
                content="Let's verify the installation by checking the versions of the installed libraries. If additional packages are required, install them directly using either `!pip install` or `!conda install` as needed.",
                role="assistant",
                name="Datawise_Agent",
            )

            c1 = CodeCell(
                content="""!grep -E '^(NAME|VERSION|ID|ID_LIKE)=' /etc/os-release
!echo "CPU Cores: $(lscpu | grep '^CPU(s):' | awk '{print $2}')"
!echo "Memory Usage: $(free -h | grep Mem | awk '{print $2 \" total, \" $3 \" used, \" $4 \" free\"}')"
!echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits 2>/dev/null || echo 'N/A')"

import importlib.metadata as md
import pandas as pd

pd.options.display.notebook_repr_html = False
for pkg in ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "torch"]:
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg} is not installed")
""",
                role="assistant",
                name="Datawise_Agent",
            )

            c2 = CodeCell(
                content="# You could use sklearn or pytorch for data modeling.\n# If you need extra packages, install them explicitly during task execution.",
                role="assistant",
                name="Datawise_Agent",
            )

            code_result = await session.safe_execute_code_blocks(c1.to_code_block())
            c1.code_output = CodeOutputCell(code_result=code_result)

            code_result = await session.safe_execute_code_blocks(c2.to_code_block())
            c2.code_output = CodeOutputCell(code_result=code_result)

            if isinstance(session.chat_history, CellHistoryMemory):
                session.chat_history.initialize(
                    [m1, m2, c1, c2, c1.code_output, c2.code_output]
                )

        else:

            m1 = MarkdownCell(
                content="# Development Environment Overview\nThe development environment is managed using `Conda` with `Python 3.10` as the interpreter. Some basic libraries for data analysis and machine learning have already been installed to streamline common tasks.",
                role="assistant",
                name="Datawise_Agent",
            )

            m2 = MarkdownCell(
                content="Let's verify the installation by checking the versions of the installed libraries. If additional packages are required, install them directly using either `!pip install` or `!conda install` as needed.",
                role="assistant",
                name="Datawise_Agent",
            )

            c1 = CodeCell(
                content="""!grep -E '^(NAME|VERSION|ID|ID_LIKE)=' /etc/os-release

import importlib.metadata as md
import pandas as pd

# disable HTML representation of dataframe
pd.options.display.notebook_repr_html = False
for pkg in ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn"]:
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg} is not installed")
""",
                role="assistant",
                name="Datawise_Agent",
            )

            c2 = CodeCell(
                content="# Environment check finished. Install additional packages explicitly when needed.",
                role="assistant",
                name="Datawise_Agent",
            )

            code_result = await session.safe_execute_code_blocks(c1.to_code_block())
            c1.code_output = CodeOutputCell(code_result=code_result)

            code_result = await session.safe_execute_code_blocks(c2.to_code_block())
            c2.code_output = CodeOutputCell(code_result=code_result)

            if tool_mode == "default":
                if isinstance(session.chat_history, CellHistoryMemory):
                    session.chat_history.initialize(
                        [m1, m2, c1, c2, c1.code_output, c2.code_output]
                    )
            elif tool_mode == "dsbench":
                import datawiseagent.tools.dsbench
                from io import BytesIO
                import inspect

                try:
                    dsbench_pkg = datawiseagent.tools.dsbench
                    package_file = inspect.getfile(dsbench_pkg)
                    package_dir = os.path.dirname(package_file)
                    print(f"dsbench package directory: {package_dir}")
                except TypeError as e:
                    print(f"Failed to retrieve dsbench package file path: {e}")
                    return

                upload_files = []
                # Traverse the dsbench directory and collect all files
                for root, dirs, files in os.walk(package_dir):
                    # Exclude the __pycache__ directory
                    if "__pycache__" in dirs:
                        dirs.remove("__pycache__")

                    for file in files:

                        # You can filter specific file types if needed, e.g., upload only .py files
                        # Use os.path.splitext to filter by file extension
                        _, ext = os.path.splitext(file)

                        # Upload only .py files, skip .pyc and other non-source files
                        if ext != ".py":
                            continue

                        full_path = os.path.join(root, file)
                        # Get the relative path to the dsbench root directory to preserve directory structure
                        relative_path = os.path.relpath(full_path, package_dir)
                        try:
                            with open(full_path, "rb") as f:
                                file_content = f.read()
                            file_like = BytesIO(file_content)
                            # Use the relative path as the filename to preserve directory structure
                            # Normalize path separators to '/' to avoid cross-platform issues
                            normalized_relative_path = relative_path.replace(
                                os.sep, "/"
                            )
                            upload_file = UploadFile(
                                filename=normalized_relative_path, file=file_like
                            )
                            upload_files.append(upload_file)
                            print(f"Added file to upload: {normalized_relative_path}")
                        except Exception as e:
                            print(f"Failed to read file {relative_path}: {e}")

                await self.process_uploaded_files(
                    session.session_id, upload_files, dir_to_save="system"
                )

                c3 = CodeCell(
                    content="# You can use the function `evaluate_image()` to access the descriptive and analytical information about any image files.\nfrom system.vision_tool import evaluate_image\nimport inspect\nsignature = inspect.signature(evaluate_image)\nprint(signature)\nprint(evaluate_image.__doc__)\n",
                    role="assistant",
                    name="Datawise_Agent",
                )
                code_result = await session.safe_execute_code_blocks(c3.to_code_block())
                c3.code_output = CodeOutputCell(code_result=code_result)

                if isinstance(session.chat_history, CellHistoryMemory):
                    session.chat_history.initialize(
                        [
                            m1,
                            m2,
                            c1,
                            c2,
                            c1.code_output,
                            c2.code_output,
                            c3,
                            c3.code_output,
                        ]
                    )

    async def stop_session(self, session_id: uuid.UUID) -> uuid.UUID:
        """
        Stop a running session and release associated resources.

        Parameters
        ----------
        session_id : uuid.UUID
            The session to stop.

        Returns
        -------
        uuid.UUID
            The same `session_id`, for convenience.

        Raises
        ------
        KeyError
            If the session does not exist.
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
            code_executor = session.code_executor
            if isinstance(code_executor, JupyterCodeExecutor):
                await code_executor.stop()
            self.sessions.pop(session_id)
        else:
            raise KeyError(f"Session ID {session_id} not found in active sessions.")

        return session_id

    async def process_uploaded_files(
        self,
        session_id: uuid.UUID,
        files: List[UploadFile],
        dir_to_save: Literal["root", "working", "display", "input", "system"] = "input",
    ):
        """
        Save uploaded files into a session-scoped directory.

        Parameters
        ----------
        session_id : uuid.UUID
            Session receiving the files.
        files : list[UploadFile]
            Files to store.
        dir_to_save : {"root","working","display","input","system"}, default="input"
            Target directory within the session workspace.

        Returns
        -------
        list[str]
            Absolute filesystem paths to the saved files.

        Raises
        ------
        KeyError
            If the session does not exist.
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            raise KeyError(f"Session ID {session_id} not found in active sessions.")

        saved_files = []
        for file in files:
            # Define the path to save the uploaded file
            if dir_to_save == "input":
                save_path = session.input_dir / file.filename
            elif dir_to_save == "system":
                save_path = session.system_dir / file.filename
            elif dir_to_save == "display":
                save_path = session.display_dir / file.filename
            elif dir_to_save == "working":
                save_path = session.working_dir / file.filename
            elif dir_to_save == "root":
                save_path = session.root_dir / file.filename

            with open(save_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(save_path)

        # You can add logic to process the content of the files here.
        return [str(path) for path in saved_files]
 

    async def _invoke_agent_with_experience(
        self,
        session_id: str,
        task_id: str,
        step_id: int,
        small_step_id: int,
        physical_agent: PhysicalAgent,
        network_latency: float,
        cell_size: int,
        agent_function,
        *args,
        source_zone: Optional[str] = None,
        target_zone: Optional[str] = None,
        cells: Optional[list] = None,
        cac_data: Optional[dict] = None,
        **kwargs,
    ):
        """一个包装器，用于调用子代理函数，并使用其唯一名称记录性能数据。"""
        max_physical_retries = 0 # !!!!!!!!!!!!!!!! can change
        base_backoff_s = 0.2
        role = ""
        agent_base_name = ""
        try:
            role, agent_base_name = physical_agent.name.split("-", 1)
        except Exception:
            role, agent_base_name = "", ""

        attempts_history = []
        physical_attempt = 0
        while True:
            physical_attempt += 1

            # This is for the link from the previous agent to this one. This should always be logged.
            await experience_collecting.update_experience_network(
                agent_from=self.last_small_step_agent,
                agent_to=physical_agent.name,
                task_id=task_id,
                network_latency=network_latency,
                cell_size=cell_size,
                network_mode=self.network_mode,
            )

            if network_latency == -1.0:
                logger.warn(
                    f"[Executor] Simulated packet loss for agent '{physical_agent.name}'. "
                    f"Attempt {physical_attempt}/{max_physical_retries + 1}."
                )
                # Log the failed attempt. agent_latency and token_count are 0.
                experience_collecting.log_task_experience(
                    session_id,
                    task_id,
                    step_id,
                    small_step_id,
                    physical_agent.name,
                    0,
                    0,
                    network_latency,
                    cell_size,
                    cac_data=cac_data,
                    failure_type="network_packet_loss",
                )
                attempts_history.append({
                    "type": "network_packet_loss",
                    "latency": 0,
                    "network_latency": network_latency
                })

                if physical_attempt <= max_physical_retries and cells is not None:
                    await asyncio.sleep(base_backoff_s * physical_attempt)
                    network_latency, cell_size = self._compute_network_latency(
                        source_zone,
                        target_zone,
                        cells,
                        role=role,
                        agent_name=agent_base_name,
                    )
                    continue

                raise ConnectionError(f"Simulated packet loss for agent {physical_agent.name}")
            # 从代理名称中解析出角色和代理名称，以便查询配置
            try:
                agent_config_detail = self.agent_llm_manager.agent_configs.get(role, {}).get(agent_base_name, {})
                # Failure injection is optional in real mode:
                # - if `real_failure_rate` is not configured, do not inject agent failure.
                # - if `real_failure_rate` is configured, use it.
                # Simulated mode keeps using `failure_rate`.
                if self.real_mode:
                    raw_failure_rate = agent_config_detail.get("real_failure_rate")
                    if raw_failure_rate is None:
                        failure_rate = 0.0
                    else:
                        failure_rate = max(0.0, min(1.0, float(raw_failure_rate)))
                else:
                    raw_failure_rate = agent_config_detail.get("failure_rate", 0.0)
                    failure_rate = max(0.0, min(1.0, float(raw_failure_rate)))
            except Exception as e:
                logger.error(f"无法获取代理 '{physical_agent.name}' 的 failure_rate: {e}")
                failure_rate = 0.0

            # 模拟代理宕机
            if random.random() < failure_rate:
                logger.warn(
                    f"[Executor] Simulated failure for agent '{physical_agent.name}' due to failure_rate: {failure_rate}. "
                    f"Attempt {physical_attempt}/{max_physical_retries + 1}."
                )
                experience_collecting.log_task_experience(
                    session_id,
                    task_id,
                    step_id,
                    small_step_id,
                    physical_agent.name,
                    0,
                    0,
                    network_latency,
                    cell_size,
                    cac_data=cac_data,
                    failure_type="agent_failure_rate",
                )
                attempts_history.append({
                    "type": "agent_failure_rate",
                    "latency": 0,
                    "network_latency": network_latency
                })

                if physical_attempt <= max_physical_retries:
                    await asyncio.sleep(base_backoff_s * physical_attempt)
                    continue

                raise ConnectionError(
                    f"Simulated failure for agent {physical_agent.name} due to agent failure_rate: {failure_rate}"
                )

            break

        # If no packet loss and no simulated failure, proceed with the actual agent call.
        candidate_runtime = kwargs.pop("_candidate_runtime", None)
        start_time = time.time()
        debug_trace = kwargs.get("debug_trace")
        try:
            result = await agent_function(physical_agent, *args, **kwargs)
        except asyncio.CancelledError:
            if candidate_runtime is not None:
                await self._close_candidate_runtime(
                    candidate_runtime,
                    cleanup_workspace=True,
                )
            raise
        except Exception:
            if candidate_runtime is not None:
                await self._close_candidate_runtime(
                    candidate_runtime,
                    cleanup_workspace=True,
                )
            raise
        end_time = time.time()
        latency = end_time - start_time

        token_count = 0
        if result and isinstance(result, tuple) and len(result) > 1:
            node = result[1]
            if isinstance(node, Node) and hasattr(node, 'completion_usage') and node.completion_usage and hasattr(node.completion_usage, 'total_tokens'):
                token_count = node.completion_usage.total_tokens
        logger.debug(f"Extracted token_count for {physical_agent.name} (task_id: {task_id}): {token_count}")

        # Log the successful attempt.
        experience_collecting.log_task_experience(session_id, task_id, step_id, small_step_id, physical_agent.name, latency, token_count, network_latency, cell_size, cac_data=cac_data)
        attempts_history.append({
            "type": "success",
            "latency": latency,
            "network_latency": network_latency
        })

        return {
            "name": physical_agent.name,
            "result": result,
            "attempts": attempts_history,
            "cac_data": cac_data,
            "debug_trace": debug_trace,
            "candidate_runtime": candidate_runtime,
        }

    @staticmethod
    def _is_session_like(obj: Any) -> bool:
        return (
            hasattr(obj, "chat_history")
            and hasattr(obj, "code_executor")
            and hasattr(obj, "workspace_root_dir")
        )

    def _replace_session_arg(
        self,
        args: tuple[Any, ...],
        session_like: Any,
    ) -> tuple[Any, ...]:
        updated_args = list(args)
        for idx, arg in enumerate(updated_args):
            if self._is_session_like(arg):
                updated_args[idx] = session_like
                break
        return tuple(updated_args)

    @staticmethod
    def _extract_generated_source_cells(node: Node) -> list[NotebookCell]:
        node_cells = (
            [node.cells_generated]
            if isinstance(node.cells_generated, NotebookCell)
            else list(node.cells_generated)
        )
        source_cells: list[NotebookCell] = []
        for cell in node_cells:
            if isinstance(cell, CodeOutputCell):
                continue
            source_cells.append(deepcopy(cell))
        return source_cells

    async def _replay_selected_result_on_main_session(
        self,
        session: Session,
        result: Any,
    ) -> Any:
        action_signal, node = self._coerce_action_node_result(
            result,
            "candidate_runtime.replay",
        )
        if node is None or not isinstance(node, Node):
            return result

        generated_cells = self._extract_generated_source_cells(node)
        rerun_all = False
        if isinstance(node, StepNode) and action_signal == ITERATE_ON_LAST_STEP:
            session.chat_history.clear_last_step()
            rerun_all = True
        elif isinstance(node, PostDebuggingNode):
            rerun_all = True

        output_cells = await execution.execute_cells(
            session,
            generated_cells,
            rerun_all=rerun_all,
        )
        replayed_cells = generated_cells + output_cells
        completion_usage = deepcopy(node.completion_usage)

        if isinstance(node, StepNode):
            replayed_node = StepNode(
                cells_generated=replayed_cells,
                action_signal=action_signal,
                completion_usage=completion_usage,
            )
        elif isinstance(node, ExecutionNode):
            replayed_node = ExecutionNode(
                cells_generated=replayed_cells,
                action_signal=action_signal,
                completion_usage=completion_usage,
            )
        elif isinstance(node, PostDebuggingNode):
            replayed_node = PostDebuggingNode(
                cells_generated=replayed_cells,
                action_signal=action_signal,
                completion_usage=completion_usage,
            )
        elif isinstance(node, DebugNode):
            replayed_node = DebugNode(
                cells_generated=replayed_cells,
                action_signal=action_signal,
                completion_usage=completion_usage,
            )
        else:
            return result

        return action_signal, replayed_node

    async def _close_candidate_runtime(
        self,
        candidate_runtime: Any,
        cleanup_workspace: bool = True,
        stop_executor: bool = True,
    ) -> None:
        if not getattr(candidate_runtime, "is_candidate_runtime", False):
            return

        await candidate_runtime.close(
            cleanup_workspace=cleanup_workspace,
            stop_executor=stop_executor,
        )

    async def _close_debug_trace_runtime(
        self,
        debug_trace: Any,
        cleanup_workspace: bool = True,
    ) -> None:
        if debug_trace is None:
            return
        session_view = getattr(debug_trace, "session_view", None)
        if session_view is None:
            return
        await self._close_candidate_runtime(
            getattr(session_view, "_base_session", None),
            cleanup_workspace=cleanup_workspace,
        )

    async def _invoke_agent_without_candidate_runtime(
        self,
        session_id: str,
        task_id: str,
        step_id: int,
        small_step_id: int,
        role: str,
        physical_agent: PhysicalAgent,
        network_latency: float,
        cell_size: int,
        agent_function,
        *args,
        source_zone: Optional[str] = None,
        target_zone: Optional[str] = None,
        cells: Optional[list] = None,
        cac_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        session = self.sessions[uuid.UUID(session_id)]
        call_kwargs = kwargs.copy()

        if role == "debugger" and call_kwargs.get("debug_trace") is None:
            logger.info(
                f"Executor: single candidate fast path for role '{role}' using main session."
            )
            call_kwargs["debug_trace"] = create_debug_trace_record(session)
        elif role == "filter" and call_kwargs.get("debug_trace") is not None:
            logger.info(
                "Executor: single candidate fast path for role 'filter' reusing "
                "existing debug trace runtime."
            )
        else:
            logger.info(
                f"Executor: single candidate fast path for role '{role}' using main session."
            )

        return await self._invoke_agent_with_experience(
            session_id,
            task_id,
            step_id,
            small_step_id,
            physical_agent,
            network_latency,
            cell_size,
            agent_function,
            *args,
            source_zone=source_zone,
            target_zone=target_zone,
            cells=cells,
            cac_data=cac_data,
            **call_kwargs,
        )

    async def _invoke_agent_with_candidate_runtime(
        self,
        session_id: str,
        task_id: str,
        step_id: int,
        small_step_id: int,
        role: str,
        physical_agent: PhysicalAgent,
        network_latency: float,
        cell_size: int,
        agent_function,
        *args,
        source_zone: Optional[str] = None,
        target_zone: Optional[str] = None,
        cells: Optional[list] = None,
        cac_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        session = self.sessions[uuid.UUID(session_id)]
        candidate_runtime = None
        incoming_debug_trace = kwargs.get("debug_trace")
        source_chat_history = None
        if role == "filter" and incoming_debug_trace is not None:
            source_chat_history = incoming_debug_trace.chat_history

        candidate_id = (
            f"{role}-{physical_agent.name}-{task_id}-{step_id}-{small_step_id}-"
            f"{uuid.uuid4().hex[:8]}"
        )

        try:
            candidate_runtime = await CandidateSessionView.create(
                session,
                candidate_id=candidate_id,
                source_chat_history=source_chat_history,
            )
            call_args = self._replace_session_arg(args, candidate_runtime)
            call_kwargs = kwargs.copy()
            if role == "debugger":
                call_kwargs["debug_trace"] = create_debug_trace_record(candidate_runtime)
            elif role == "filter" and incoming_debug_trace is not None:
                call_kwargs["debug_trace"] = clone_debug_trace_record(
                    incoming_debug_trace,
                    base_session=candidate_runtime,
                )

            return await self._invoke_agent_with_experience(
                session_id,
                task_id,
                step_id,
                small_step_id,
                physical_agent,
                network_latency,
                cell_size,
                agent_function,
                *call_args,
                source_zone=source_zone,
                target_zone=target_zone,
                cells=cells,
                cac_data=cac_data,
                _candidate_runtime=candidate_runtime,
                **call_kwargs,
            )
        except Exception:
            if candidate_runtime is not None:
                await self._close_candidate_runtime(
                    candidate_runtime,
                    cleanup_workspace=True,
                )
            raise

    async def _invoke_agents_concurrently(self, session_id: str, task_id: str, step_id: int, small_step_id: int, role: str, agent_function, *args, **kwargs) -> Any:
        """
        Invokes multiple agents concurrently based on a router's strategy.
        It executes agents, gathers results according to the EXECUTION_STRATEGY,
        and uses the router's process_results to return a single final result.
        """
        return_selected_payload = kwargs.pop("_return_selected_payload", False)

        # 1. Load Router
        router_name = self.agent_llm_manager.selection_strategy
        try:
            module_path = f"datawiseagent.agents.os.process_management.{router_name}"
            router_module = importlib.import_module(module_path)
            logger.info(f"Executor: Loaded router '{router_name}'.")
        except (ImportError, AttributeError) as e:
            logger.error(f"Executor: Failed to load router module '{router_name}'. Error: {e}. Aborting.")
            # Fallback or raise an exception
            return None 

        # 2. Get Agent Configurations
        get_agents_start_time = time.time()
        agent_configs = []
        cac_params = None
        if router_name == 'cac_router' or router_name.startswith('haco'):
            agent_configs, cac_params = router_module.get_selected_agent_list(role)
        else:
            agent_configs = router_module.get_selected_agent_list(role)
        
        get_agents_end_time = time.time()
        selection_latency = get_agents_end_time - get_agents_start_time
        
        if not agent_configs:
            logger.warn(f"Executor: Router '{router_name}' returned no agents for role '{role}'.")
            # If no agents, log router experience with just selection_latency
            experience_collecting.log_router_experience(session_id, task_id, step_id, small_step_id, selection_latency, selected_agent=None, call_count=0)
            return None
        # 3. Read Execution Strategy from Router
        exec_strategy = getattr(router_module, "EXECUTION_STRATEGY", "wait_for_all") # Default to safe "wait_for_all"
        logger.info(f"Executor: Using execution strategy '{exec_strategy}' from router '{router_name}'.")

        # 4. Create Agent Tasks
        tasks = []
        single_candidate_fast_path = len(agent_configs) == 1
        for agent_info in agent_configs:
            llm_client = load_llm(deepcopy(agent_info["config"]))
            agent_name = f"{role}-{agent_info['name']}"
            physical_agent = PhysicalAgent(name=agent_name, llm_client=llm_client)
            
            # Calculate network latency and cell size before creating the task
            source_zone = self.last_small_step_agent_zone
            target_zone = self._get_agent_zone(role, agent_info.get("name"))
            session = self.sessions[uuid.UUID(session_id)]
            cells = session.chat_history.cells
            network_latency, cell_size = self._compute_network_latency(
                source_zone,
                target_zone,
                cells,
                role=role,
                agent_name=agent_info.get("name"),
            )
            logger.debug(f"Calculated network_latency for agent {agent_name} from {source_zone} to {target_zone}: {network_latency:.4f}s, cell_size: {cell_size} bytes")
            
            cac_data_for_agent = cac_params.get(agent_name) if cac_params else None

            # 创建一个新的 kwargs 字典来安全地包含 cac_data。
            # 这不会影响 moa 的选择逻辑，只是调整了参数传递机制。
            call_kwargs = kwargs.copy()

            if single_candidate_fast_path:
                task = asyncio.create_task(
                    self._invoke_agent_without_candidate_runtime(
                        session_id,
                        task_id,
                        step_id,
                        small_step_id,
                        role,
                        physical_agent,
                        network_latency,
                        cell_size,
                        agent_function,
                        *args,
                        source_zone=source_zone,
                        target_zone=target_zone,
                        cells=cells,
                        cac_data=cac_data_for_agent,
                        **call_kwargs,
                    )
                )
            else:
                task = asyncio.create_task(
                    self._invoke_agent_with_candidate_runtime(
                        session_id,
                        task_id,
                        step_id,
                        small_step_id,
                        role,
                        physical_agent,
                        network_latency,
                        cell_size,
                        agent_function,
                        *args,
                        source_zone=source_zone,
                        target_zone=target_zone,
                        cells=cells,
                        cac_data=cac_data_for_agent,
                        **call_kwargs  # 使用包含 cac_data 的新字典
                    )
                )
            tasks.append(task)
        
        # 5. Execute Agents based on Strategy
        raw_results = []
        if exec_strategy == "fastest_first":
            if not tasks: 
                experience_collecting.log_router_experience(session_id, task_id, step_id, small_step_id, selection_latency)
                return None
            
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Gather results from the completed tasks
            for task in done:
                try:
                    raw_results.append(task.result())
                except Exception as e:
                    raw_results.append(e) # Store exception as a result
            
            # Handle pending tasks in the background for experience logging
            if pending:
                async def _handle_pending():
                    logger.info(f"Executor: Handling {len(pending)} slow agents in the background for role '{role}'.")
                    background_results = await asyncio.gather(
                        *pending,
                        return_exceptions=True,
                    )
                    for payload in background_results:
                        if isinstance(payload, dict):
                            await self._close_candidate_runtime(
                                payload.get("candidate_runtime"),
                                cleanup_workspace=True,
                            )
                
                asyncio.create_task(_handle_pending())

        else: # "wait_for_all" or default
            done, pending = await asyncio.wait(tasks, timeout=300)

            raw_results = []
            for task in done:
                try:
                    raw_results.append(task.result())
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(f"Executor: Task for role '{role}' failed with exception: {e}", f"TRACEBACK:\n{tb_str}")
                    raw_results.append(e)

            if pending:
                for task in pending:
                    task.cancel()
                    logger.warn(f"Executor: A task for role '{role}' timed out after 5 minutes and was cancelled.")
                    raw_results.append(asyncio.TimeoutError("Task timed out after 300 seconds"))

        # 6. Pass results to Router for processing
        process_results_start_time = time.time()
        logger.info(f"Executor: Passing {len(raw_results)} result(s) to router for processing.")
        
        # New conditional logic for MoA router
        router_network_latency = 0.0
        if router_name == "moa":
            router_network_latency = await self._calculate_max_agent_to_router_latency(raw_results, role)

        # Check if the router's process_results function expects a 'query' argument
        sig = inspect.signature(router_module.process_results)
        router_token_count = 0
        if 'query' in sig.parameters:
            session = self.sessions[uuid.UUID(session_id)]
            # Refactored: Create a context-aware query from recent chat history
            # instead of just using the initial user query. This provides the evaluator
            # with the immediate context of the current task.
            context_cells = []
            # Iterate in reverse to get the most recent and relevant cells.
            # Only collect UserCell (user instructions/queries) to avoid polluting
            # the judge query with agent-generated StepCell/MarkdownCell outputs.
            for cell in reversed(session.chat_history.cells):
                if isinstance(cell, UserCell):
                    context_cells.append(content_str(cell.content))
                # Stop after collecting a few context items to keep it relevant.
                if len(context_cells) >= 3:
                    break
            
            # The context should be in chronological order for the LLM
            context_query = "\n\n---\n\n".join(reversed(context_cells))

            # If context is empty for some reason, fall back to the original user query
            if not context_query:
                context_query = session.current_user_query

            # Build step_info string for the judge, so it knows this is an intermediate step.
            step_info_parts = [f"role={role}"]
            if task_id:
                step_info_parts.append(f"task_id={task_id}")
            if step_id:
                step_info_parts.append(f"step_id={step_id}")
            if small_step_id:
                step_info_parts.append(f"small_step_id={small_step_id}")
            step_info_str = " | ".join(step_info_parts)

            judge_context = None
            if 'judge_context' in sig.parameters:
                last_step_cell = session.chat_history.find_last_step()
                current_step_goal = ""
                if isinstance(last_step_cell, NotebookCell):
                    current_step_goal = content_str(last_step_cell.content)

                history_summary = ""
                if isinstance(session.chat_history, CellHistoryMemory):
                    try:
                        _, last_response_content = session.chat_history.fetch_response()
                        history_summary = last_response_content.strip()
                    except Exception:
                        history_summary = ""

                previous_winner_result = None
                if (
                    session.last_selected_result is not None
                    and session.last_selected_step_id is not None
                    and step_id is not None
                    and session.last_selected_step_id < step_id
                ):
                    previous_winner_result = session.last_selected_result

                judge_context = {
                    "current_step_goal": current_step_goal,
                    "history_summary": history_summary,
                    "previous_winner_result": previous_winner_result,
                }
                if role in {"coder", "planner"}:
                    judge_context["redundancy_hint"] = (
                        "Compare the current response against the most recent committed result. "
                        "Penalize unnecessary repetition, especially if the step is already complete."
                    )

            router_kwargs = {
                "query": context_query,
            }
            if 'step_info' in sig.parameters:
                router_kwargs["step_info"] = step_info_str
            if judge_context is not None:
                router_kwargs["judge_context"] = judge_context

            router_result = await router_module.process_results(raw_results, role, **router_kwargs)
        else:
            router_result = await router_module.process_results(raw_results, role)

        # Unpack router result, now expecting token count
        if len(router_result) == 4:
            final_result, agent_name, agent_score_list, router_token_count = router_result
        else:
            final_result, agent_name, agent_score_list = router_result
            router_token_count = 0 # Default to 0 if not provided

        selected_payload = None
        if agent_name is not None:
            for payload in raw_results:
                if isinstance(payload, dict) and payload.get("name") == agent_name:
                    selected_payload = payload
                    break

        process_results_end_time = time.time()
        aggregation_latency = process_results_end_time - process_results_start_time

        base_router_latency = selection_latency + aggregation_latency
        # Router latency field is strategy-aware online-path latency:
        # - cac/random: selection only
        # - bestone/moa(and others): selection + aggregation
        if router_name in ("cac_router", "random") or router_name.startswith("haco"):
            current_router_latency = selection_latency
        else:
            current_router_latency = base_router_latency
        router_failure_type = None
        if isinstance(agent_score_list, list):
            has_parse_failed = any(
                isinstance(item, dict) and item.get("reasoning") == "parse_failed"
                for item in agent_score_list
            )
            if has_parse_failed:
                router_failure_type = "json_parse_failed"
        experience_collecting.log_router_experience(
            session_id, 
            task_id, 
            step_id, 
            small_step_id, 
            current_router_latency,
            selected_agent=agent_name,
            agent_scores=agent_score_list,
            router_token_count=router_token_count,
            network_latency=router_network_latency,
            role=role,
            call_count=len(agent_configs),
            selection_latency=selection_latency,
            aggregation_latency=aggregation_latency,
            failure_type=router_failure_type,
        )

        session = self.sessions[uuid.UUID(session_id)]
        selected_candidate_runtime = None
        if isinstance(selected_payload, dict):
            selected_candidate_runtime = selected_payload.get("candidate_runtime")

        try:
            if (
                selected_candidate_runtime is not None
                and role in {"planner", "coder", "filter"}
                and agent_name is not None
                and not isinstance(final_result, Exception)
            ):
                final_result = await self._replay_selected_result_on_main_session(
                    session,
                    final_result,
                )
        finally:
            for payload in raw_results:
                if not isinstance(payload, dict):
                    continue
                candidate_runtime = payload.get("candidate_runtime")
                if candidate_runtime is None or candidate_runtime is selected_candidate_runtime:
                    continue
                await self._close_candidate_runtime(
                    candidate_runtime,
                    cleanup_workspace=True,
                )

            if selected_candidate_runtime is not None and not (
                role == "debugger" and return_selected_payload
            ):
                await self._close_candidate_runtime(
                    selected_candidate_runtime,
                    cleanup_workspace=True,
                )

        if (
            agent_name is not None
            and not isinstance(final_result, Exception)
            and step_id is not None
        ):
            session.last_selected_result = final_result
            session.last_selected_step_id = step_id
        
        self.last_small_step_agent = agent_name
        
        # Use the new helper method to get the zone
        zone = self._get_agent_zone(role, agent_name)
        if zone:
            self.last_small_step_agent_zone = zone

        # 7. Return the single, final result chosen by the router
        if return_selected_payload:
            return final_result, selected_payload
        return final_result

    @staticmethod
    def _is_physical_failure_exception(err: Exception) -> bool:
        return isinstance(err, (ConnectionError, TimeoutError, asyncio.TimeoutError))

    @staticmethod
    def _coerce_action_node_result(result: Any, stage: str) -> tuple[Any, Any]:
        if isinstance(result, (tuple, list)):
            if len(result) >= 2:
                return result[0], result[1]
            logger.warn(
                f"Concurrent invocation returned too few values for stage '{stage}': "
                f"len={len(result)}. Falling back to (None, None)."
            )
            return None, None
        if result is None:
            return None, None
        logger.warn(
            f"Concurrent invocation returned unexpected payload type for stage '{stage}': "
            f"type={type(result).__name__}. Falling back to (None, None)."
        )
        return None, None

    @staticmethod
    def _coerce_agent_invocation_payload(payload: Any, stage: str) -> tuple[Any, Any]:
        if isinstance(payload, dict):
            return payload.get("name"), payload.get("result")
        if isinstance(payload, (tuple, list)) and len(payload) >= 2:
            return payload[0], payload[1]
        if payload is None:
            return None, None
        logger.warn(
            f"Single-agent invocation returned unexpected payload type for stage '{stage}': "
            f"type={type(payload).__name__}. Falling back to (None, None)."
        )
        return None, None

    async def _invoke_agents_concurrently_with_retry(
        self,
        session_id: str,
        task_id: str,
        step_id: int,
        small_step_id: int,
        role: str,
        agent_function,
        *args,
        max_stage_retries: int = 1,
        **kwargs,
    ) -> tuple[Any, bool]:
        physical_failure_observed = False
        for attempt in range(max_stage_retries + 1):
            try:
                result = await self._invoke_agents_concurrently(
                    session_id, task_id, step_id, small_step_id, role, agent_function, *args, **kwargs
                )
            except Exception as e:
                result = e

            primary_result = result
            if (
                isinstance(result, (tuple, list))
                and len(result) >= 1
                and isinstance(result[0], Exception)
            ):
                primary_result = result[0]

            if isinstance(primary_result, Exception):
                if self._is_physical_failure_exception(primary_result):
                    physical_failure_observed = True
                    if attempt < max_stage_retries:
                        logger.warn(
                            f"Concurrent invocation for role '{role}' failed with physical failure "
                            f"(attempt {attempt + 1}/{max_stage_retries + 1}), retrying."
                        )
                        continue
                return primary_result, physical_failure_observed

            return result, physical_failure_observed

        return ConnectionError(f"Concurrent invocation failed for role '{role}'"), physical_failure_observed

    async def _calculate_max_agent_to_router_latency(self, raw_results: list, role: str) -> float:
        """
        Calculates the maximum network latency from the agents that returned results to the router (local).
        This is based on the size of the results (payload).
        """
        latencies = []
        
        # The router's zone is the local orchestrator
        router_zone = "Local" 

        for res in raw_results:
            if isinstance(res, Exception):
                continue

            agent_name, agent_result = self._coerce_agent_invocation_payload(
                res,
                f"{role}.router_latency",
            )
            if not agent_name:
                continue

            _, node = self._coerce_action_node_result(
                agent_result,
                f"{role}.router_latency",
            )
            
            # We need the node and the cells it generated to calculate payload size
            if not (isinstance(node, Node) and hasattr(node, 'cells_generated')):
                continue

            # Extract agent's base name to look up its zone
            # e.g., 'planner-plan_local' -> 'plan_local'
            try:
                agent_base_name = agent_name.split(f'{role}-', 1)[1]
            except IndexError:
                logger.warn(f"Could not determine base name for agent '{agent_name}' and role '{role}'. Skipping latency calculation.")
                continue
                
            agent_zone = self._get_agent_zone(role, agent_base_name)
            
            # The payload is the cells generated by the agent
            payload_cells = node.cells_generated
            
            # Calculate latency for this specific result
            latency, _ = self._compute_network_latency(
                agent_zone,
                router_zone,
                payload_cells,
                role="router",
                agent_name="router",
            )
            latencies.append(latency)
            logger.debug(f"Calculated agent-to-router latency for {agent_name}: {latency:.4f}s")

        return max(latencies) if latencies else 0.0

    def _get_agent_zone(self, role: str, agent_name: str) -> Optional[str]:
        """
        Retrieves the zone for a given agent and role from the agent configurations.
        Returns None if the zone is not found or an error occurs.
        """
        if not agent_name:
            return None
        try:
            agent_config = self.agent_llm_manager.agent_configs.get(role, {}).get(agent_name, {})
            zone = agent_config.get('zone')
            if not zone:
                logger.warn(f"Zone not found for agent '{agent_name}' in role '{role}'.")
            return zone
        except Exception as e:
            logger.error(f"Error looking up zone for agent '{agent_name}' in role '{role}': {e}")
            return None

    async def chat(
            self,
            session_id: uuid.UUID,
            query: str,
            agent_config: dict = agent_config,
            work_mode: Literal["jupyter", "jupyter+script"] = "jupyter",
            use_moa_router: bool = False,
        ):
            DatawiseMASConcurrent._global_task_counter += 1
            task_id = str(DatawiseMASConcurrent._global_task_counter) # Use sequential task_id
            small_step_cnt = 0
            logger.info(f"--- CHAT METHOD ENTRY --- task_id: {task_id}, received session_id: {session_id}")
            
            # 从配置中动态获取 router 的名字
            router_name = list(self.agent_llm_manager.agent_configs.get('router', {}).keys())[0] if self.agent_llm_manager.agent_configs.get('router') else "router"
            self.last_small_step_agent = router_name
            self.last_small_step_agent_zone = "Local"
            
            datawise_agent_config = DatawiseAgentConfig(**agent_config)
            session: Session = self.sessions[session_id]
            session.agent_config = datawise_agent_config

            if work_mode == "jupyter":
                user_msg = LLMResult(role="user", name="USER", content=query)
            elif work_mode == "jupyter+script":
                query = query + HYBRID_WORKFLOW
                user_msg = LLMResult(role="user", name="USER", content=query)

            user_cell = NotebookCell.llm_result_convert(
                user_msg, parse_mode=ConvertType.CONVERT_USER_CELL
            )
            user_node = UserNode(cells_generated=user_cell)
            session.current_user_query = query
            await self._update_messages(session, user_node)
            session.last_selected_result = None
            session.last_selected_step_id = None
            observed_physical_failure = False

            # ---------------------------------------------------------
            # 分支 A: 开启规划模式 (FSM 架构)
            # ---------------------------------------------------------
            if session.agent_config.plan.planning:
                class State(Enum):
                    INITIATE_PLAN = "initiate_plan"
                    APPEND_CODE = "append_code"
                    DEBUG = "debug"
                    FILTER = "filter"
                    UPDATE_PLAN = "update_plan"
                    TERMINATE = "terminate"

                step_cnt = 0
                planning_cnt = 1
                append_code_cnt = 0
                debug_cnt = 0
                debug_failure_per_step = 0
                max_debug_capacity = 100
                
                state = State.INITIATE_PLAN
                action_signal = None
                current_failed_node = None
                current_debug_trace = None

                logger.info(f"$$$$$$$$$$$$$$ --- FSM starting for task_id: {task_id}, session_id: {session_id}, initial state: {state.value} ---")
                
                while not (planning_cnt >= session.agent_config.plan.planning_max_number):
                    step_cnt += 1
                    if (session.agent_config.max_step_number is not None 
                        and step_cnt >= session.agent_config.max_step_number):
                        break

                    # 判定当前是否允许自调试
                    is_self_debug = (
                        session.agent_config.debug.self_debug
                        and debug_cnt < max_debug_capacity
                        and (
                            (
                                session.agent_config.max_debug_by_step is not None
                                and debug_failure_per_step <= session.agent_config.max_debug_by_step
                            )
                            or session.agent_config.max_debug_by_step is None
                        )
                    )

                    if state == State.INITIATE_PLAN:
                        logger.info(f"$$$$$$$$$$$$$$ FSM State: {state.value} (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt+1})")
                        append_code_cnt = 0
                        debug_failure_per_step = 0
                        
                        small_step_cnt += 1
                        if use_moa_router:
                            logger.info(f"$$$$$$$$$$$$$$ Invoking agents concurrently for role 'planner' (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt})")
                            result, has_physical_failure = await self._invoke_agents_concurrently_with_retry(
                                str(session.session_id), task_id, step_cnt, small_step_cnt, 'planner', planner.initiate_step, self, session,
                                planner_task_id=task_id, planner_step_id=step_cnt, planner_small_step_id=small_step_cnt
                            )
                            if has_physical_failure:
                                observed_physical_failure = True
                            if isinstance(result, Exception):
                                logger.error(f"Planner initiation failed: {result}")
                                action_signal, step_node = (None, None) # 处理异常，将变量设置为 None
                            else:
                                action_signal, step_node = self._coerce_action_node_result(
                                    result,
                                    "planner.initiate_step",
                                )
                        else:
                            planner_agent = PhysicalAgent.get_physical_agent('planner')
                            agent_base_name = planner_agent.name.split('planner-', 1)[1]
                            target_zone = self._get_agent_zone('planner', agent_base_name)
                            source_zone = self.last_small_step_agent_zone
                            cells = session.chat_history.cells
                            network_latency, cell_size = self._compute_network_latency(
                                source_zone,
                                target_zone,
                                cells,
                                role="planner",
                                agent_name=agent_base_name,
                            )
                            logger.debug(f"Calculated network_latency for agent {planner_agent.name} from {source_zone} to {target_zone}: {network_latency:.4f}s, cell_size: {cell_size} bytes")
                            logger.info(f"$$$$$$$$$$$$$$  Invoking single agent '{planner_agent.name}' for role 'planner' (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt})")
                            agent_payload = await self._invoke_agent_with_experience(
                                str(session.session_id),
                                task_id,
                                step_cnt,
                                small_step_cnt,
                                planner_agent,
                                network_latency,
                                cell_size,
                                planner.initiate_step,
                                self,
                                session,
                                planner_task_id=task_id,
                                planner_step_id=step_cnt,
                                planner_small_step_id=small_step_cnt,
                                source_zone=source_zone,
                                target_zone=target_zone,
                                cells=cells,
                            )
                            agent_name, agent_result = self._coerce_agent_invocation_payload(
                                agent_payload,
                                "planner.initiate_step.single",
                            )
                            action_signal, step_node = self._coerce_action_node_result(
                                agent_result,
                                "planner.initiate_step.single",
                            )
                            self.last_small_step_agent = agent_name
                            if agent_name:
                                agent_base_name = agent_name.split('planner-', 1)[1]
                                self.last_small_step_agent_zone = self._get_agent_zone('planner', agent_base_name)
                        await self._update_messages(session, step_node, action_signal=action_signal)

                        if not step_node or (not execution.post_code_verification(step_node.cells_generated) and is_self_debug):
                            current_failed_node = step_node
                            state = State.DEBUG
                        else:
                            state = State.APPEND_CODE

                    elif state == State.APPEND_CODE:
                        logger.info(f"$$$$$$$$$$$$$$  FSM State: {state.value} (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt+1})")
                        append_code_cnt += 1
                        if append_code_cnt > session.agent_config.execution.execution_max_number:
                            state = State.UPDATE_PLAN
                            continue

                        small_step_cnt += 1
                        if use_moa_router:
                            logger.info(f"$$$$$$$$$$$$$$  Invoking agents concurrently for role 'coder' (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt})")
                            result, has_physical_failure = await self._invoke_agents_concurrently_with_retry(
                                str(session.session_id), task_id, step_cnt, small_step_cnt, 'coder', code.append_code, self, session, has_plan=True
                            )
                            if has_physical_failure:
                                observed_physical_failure = True
                            if isinstance(result, Exception):
                                logger.error(f"Coder execution failed: {result}")
                                action_signal, exec_node = (None, None) # 处理异常，将变量设置为 None
                            else:
                                action_signal, exec_node = self._coerce_action_node_result(
                                    result,
                                    "coder.append_code",
                                )
                        else:
                            coder_agent = PhysicalAgent.get_physical_agent('coder')
                            agent_base_name = coder_agent.name.split('coder-', 1)[1]
                            target_zone = self._get_agent_zone('coder', agent_base_name)
                            source_zone = self.last_small_step_agent_zone
                            cells = session.chat_history.cells
                            network_latency, cell_size = self._compute_network_latency(
                                source_zone,
                                target_zone,
                                cells,
                                role="coder",
                                agent_name=agent_base_name,
                            )
                            logger.debug(f"Calculated network_latency for agent {coder_agent.name} from {source_zone} to {target_zone}: {network_latency:.4f}s, cell_size: {cell_size} bytes")
                            logger.info(f"$$$$$$$$$$$$$$  Invoking single agent '{coder_agent.name}' for role 'coder' (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt})")
                            agent_payload = await self._invoke_agent_with_experience(
                                str(session.session_id),
                                task_id,
                                step_cnt,
                                small_step_cnt,
                                coder_agent,
                                network_latency,
                                cell_size,
                                code.append_code,
                                self,
                                session,
                                has_plan=True,
                                source_zone=source_zone,
                                target_zone=target_zone,
                                cells=cells,
                            )
                            agent_name, agent_result = self._coerce_agent_invocation_payload(
                                agent_payload,
                                "coder.append_code.single",
                            )
                            action_signal, exec_node = self._coerce_action_node_result(
                                agent_result,
                                "coder.append_code.single",
                            )
                            self.last_small_step_agent = agent_name
                            if agent_name:
                                agent_base_name = agent_name.split('coder-', 1)[1]
                                self.last_small_step_agent_zone = self._get_agent_zone('coder', agent_base_name)
                        await self._update_messages(session, exec_node, action_signal=action_signal)

                        if not exec_node or (not execution.post_code_verification(exec_node.cells_generated) and is_self_debug):
                            current_failed_node = exec_node
                            state = State.DEBUG
                        else:
                            if action_signal == AWAIT_TAG:
                                state = State.APPEND_CODE
                            else:
                                state = State.UPDATE_PLAN

                    elif state == State.DEBUG:
                        logger.info(f"$$$$$$$$$$$$$$  FSM State: {state.value} (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt+1})")
                        debug_cnt += 1
                        debug_failure_per_step += 1

                        if current_failed_node is None:
                            logger.warn("DEBUG state entered with None current_failed_node. Skipping debugger and continuing planning.")
                            state = State.UPDATE_PLAN
                            continue

                        debug.filter_safe_cells(session, current_failed_node)

                        small_step_cnt += 1
                        if use_moa_router:
                            current_debug_trace = None
                            logger.info(f"$$$$$$$$$$$$$$  Invoking agents concurrently for role 'debugger' (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt})")
                            debug_payload, has_physical_failure = await self._invoke_agents_concurrently_with_retry(
                                str(session.session_id),
                                task_id,
                                step_cnt,
                                small_step_cnt,
                                'debugger',
                                debug.self_debug_execution,
                                self,
                                session,
                                session.agent_config.debug.debug_max_number,
                                has_plan=True,
                                _return_selected_payload=True,
                            )
                            if has_physical_failure:
                                observed_physical_failure = True
                            if isinstance(debug_payload, Exception):
                                logger.error(f"Debugger execution failed: {debug_payload}")
                            else:
                                _, selected_payload = (
                                    debug_payload
                                    if isinstance(debug_payload, tuple)
                                    and len(debug_payload) == 2
                                    else (debug_payload, None)
                                )
                                if isinstance(selected_payload, dict):
                                    current_debug_trace = selected_payload.get("debug_trace")
                        else:
                            debugger_agent = PhysicalAgent.get_physical_agent('debugger')
                            agent_base_name = debugger_agent.name.split('debugger-', 1)[1]
                            target_zone = self._get_agent_zone('debugger', agent_base_name)
                            source_zone = self.last_small_step_agent_zone
                            cells = session.chat_history.cells
                            network_latency, cell_size = self._compute_network_latency(
                                source_zone,
                                target_zone,
                                cells,
                                role="debugger",
                                agent_name=agent_base_name,
                            )
                            logger.debug(f"Calculated network_latency for agent {debugger_agent.name} from {source_zone} to {target_zone}: {network_latency:.4f}s, cell_size: {cell_size} bytes")
                            logger.info(f"$$$$$$$$$$$$$$  Invoking single agent '{debugger_agent.name}' for role 'debugger' (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt})")
                            current_debug_trace = create_debug_trace_record(session)
                            agent_payload = await self._invoke_agent_with_experience(
                                str(session.session_id),
                                task_id,
                                step_cnt,
                                small_step_cnt,
                                debugger_agent,
                                network_latency,
                                cell_size,
                                debug.self_debug_execution,
                                self,
                                session,
                                session.agent_config.debug.debug_max_number,
                                has_plan=True,
                                debug_trace=current_debug_trace,
                                source_zone=source_zone,
                                target_zone=target_zone,
                                cells=cells,
                            )
                            agent_name, _ = self._coerce_agent_invocation_payload(
                                agent_payload,
                                "debug.self_debug_execution.single",
                            )
                            self.last_small_step_agent = agent_name
                            if agent_name:
                                agent_base_name = agent_name.split('debugger-', 1)[1]
                                self.last_small_step_agent_zone = self._get_agent_zone('debugger', agent_base_name)
                        state = State.FILTER

                    elif state == State.FILTER:
                        logger.info(f"$$$$$$$$$$$$$$  FSM State: {state.value} (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt+1})")
                        small_step_cnt += 1
                        if use_moa_router:
                            logger.info(f"$$$$$$$$$$$$$$  Invoking agents concurrently for role 'filter' (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt})")
                            if current_debug_trace is not None:
                                try:
                                    filter_result, has_physical_failure = await self._invoke_agents_concurrently_with_retry(
                                        str(session.session_id),
                                        task_id,
                                        step_cnt,
                                        small_step_cnt,
                                        'filter',
                                        filter.self_debug_filter,
                                        self,
                                        session,
                                        has_plan=True,
                                        debug_trace=current_debug_trace,
                                    )
                                    if has_physical_failure:
                                        observed_physical_failure = True
                                    if isinstance(filter_result, Exception):
                                        logger.error(f"Filter execution failed: {filter_result}")
                                    else:
                                        filter_action_signal, filter_node = self._coerce_action_node_result(
                                            filter_result,
                                            "filter.self_debug_filter.concurrent",
                                        )
                                        if filter_node is not None:
                                            await self._update_messages(
                                                session,
                                                filter_node,
                                                action_signal=filter_action_signal,
                                            )
                                        current_failed_node = None
                                finally:
                                    await self._close_debug_trace_runtime(
                                        current_debug_trace,
                                        cleanup_workspace=True,
                                    )
                                    current_debug_trace = None
                            else:
                                filter_result, has_physical_failure = await self._invoke_agents_concurrently_with_retry(
                                    str(session.session_id),
                                    task_id,
                                    step_cnt,
                                    small_step_cnt,
                                    'filter',
                                    filter.self_debug_filter,
                                    self,
                                    session,
                                    has_plan=True,
                                )
                                if has_physical_failure:
                                    observed_physical_failure = True
                                if isinstance(filter_result, Exception):
                                    logger.error(f"Filter execution failed: {filter_result}")
                        else:
                            filter_agent = PhysicalAgent.get_physical_agent('filter')
                            agent_base_name = filter_agent.name.split('filter-', 1)[1]
                            target_zone = self._get_agent_zone('filter', agent_base_name)
                            source_zone = self.last_small_step_agent_zone
                            cells = session.chat_history.cells
                            network_latency, cell_size = self._compute_network_latency(
                                source_zone,
                                target_zone,
                                cells,
                                role="filter",
                                agent_name=agent_base_name,
                            )
                            logger.debug(f"Calculated network_latency for agent {filter_agent.name} from {source_zone} to {target_zone}: {network_latency:.4f}s, cell_size: {cell_size} bytes")
                            logger.info(f"$$$$$$$$$$$$$$  Invoking single agent '{filter_agent.name}' for role 'filter' (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt})")
                            if current_debug_trace is None:
                                current_debug_trace = create_debug_trace_record(session)
                            agent_payload = await self._invoke_agent_with_experience(
                                str(session.session_id),
                                task_id,
                                step_cnt,
                                small_step_cnt,
                                filter_agent,
                                network_latency,
                                cell_size,
                                filter.self_debug_filter,
                                self,
                                session,
                                has_plan=True,
                                debug_trace=current_debug_trace,
                                source_zone=source_zone,
                                target_zone=target_zone,
                                cells=cells,
                            )
                            agent_name, filter_result = self._coerce_agent_invocation_payload(
                                agent_payload,
                                "filter.self_debug_filter.single",
                            )
                            filter_action_signal, filter_node = self._coerce_action_node_result(
                                filter_result,
                                "filter.self_debug_filter.single",
                            )
                            self.last_small_step_agent = agent_name
                            if agent_name:
                                agent_base_name = agent_name.split('filter-', 1)[1]
                                self.last_small_step_agent_zone = self._get_agent_zone('filter', agent_base_name)
                            if filter_node is not None:
                                await self._update_messages(
                                    session,
                                    filter_node,
                                    action_signal=filter_action_signal,
                                )
                            await self._close_debug_trace_runtime(
                                current_debug_trace,
                                cleanup_workspace=True,
                            )
                            current_debug_trace = None
                            current_failed_node = None
                        
                        state = State.APPEND_CODE

                    elif state == State.UPDATE_PLAN:
                        logger.info(f"$$$$$$$$$$$$$$  FSM State: {state.value} (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt+1})")
                        append_code_cnt = 0
                        debug_failure_per_step = 0

                        small_step_cnt += 1
                        if use_moa_router:
                            logger.info(f"$$$$$$$$$$$$$$  Invoking agents concurrently for role 'planner' (update plan) (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt})")
                            result, has_physical_failure = await self._invoke_agents_concurrently_with_retry(
                                str(session.session_id), task_id, step_cnt, small_step_cnt, 'planner', planner.update_plan, self, session,
                                planner_task_id=task_id, planner_step_id=step_cnt, planner_small_step_id=small_step_cnt
                            )
                            if has_physical_failure:
                                observed_physical_failure = True
                            if isinstance(result, Exception):
                                logger.error(f"Planner update failed: {result}")
                                action_signal, step_node = (None, None) # 处理异常，将变量设置为 None
                            else:
                                action_signal, step_node = self._coerce_action_node_result(
                                    result,
                                    "planner.update_plan",
                                )
                        else:
                            planner_agent = PhysicalAgent.get_physical_agent('planner')
                            agent_base_name = planner_agent.name.split('planner-', 1)[1]
                            target_zone = self._get_agent_zone('planner', agent_base_name)
                            source_zone = self.last_small_step_agent_zone
                            cells = session.chat_history.cells
                            network_latency, cell_size = self._compute_network_latency(
                                source_zone,
                                target_zone,
                                cells,
                                role="planner",
                                agent_name=agent_base_name,
                            )
                            logger.debug(f"Calculated network_latency for agent {planner_agent.name} from {source_zone} to {target_zone}: {network_latency:.4f}s, cell_size: {cell_size} bytes")
                            logger.info(f"$$$$$$$$$$$$$$  Invoking single agent '{planner_agent.name}' for role 'planner' (update plan) (step_cnt: {step_cnt}, small_step_cnt: {small_step_cnt})")
                            agent_payload = await self._invoke_agent_with_experience(
                                str(session.session_id),
                                task_id,
                                step_cnt,
                                small_step_cnt,
                                planner_agent,
                                network_latency,
                                cell_size,
                                planner.update_plan,
                                self,
                                session,
                                planner_task_id=task_id,
                                planner_step_id=step_cnt,
                                planner_small_step_id=small_step_cnt,
                                source_zone=source_zone,
                                target_zone=target_zone,
                                cells=cells,
                            )
                            agent_name, agent_result = self._coerce_agent_invocation_payload(
                                agent_payload,
                                "planner.update_plan.single",
                            )
                            action_signal, step_node = self._coerce_action_node_result(
                                agent_result,
                                "planner.update_plan.single",
                            )
                            self.last_small_step_agent = agent_name
                            if agent_name:
                                agent_base_name = agent_name.split('planner-', 1)[1]
                                self.last_small_step_agent_zone = self._get_agent_zone('planner', agent_base_name)
                        await self._update_messages(session, step_node, action_signal=action_signal)

                        if action_signal in [ITERATE_ON_LAST_STEP, ADVANCE_TO_NEXT_STEP]:
                            planning_cnt += 1
                            if not step_node or (not execution.post_code_verification(step_node.cells_generated) and is_self_debug):
                                current_failed_node = step_node
                                state = State.DEBUG
                            else:
                                state = State.APPEND_CODE
                        elif action_signal == FULFILL_INSTRUCTION:
                            state = State.TERMINATE
                        else:
                            state = State.TERMINATE

                    elif state == State.TERMINATE:
                        logger.info(f"$$$$$$$$$$$$$$ FSM State: {state.value} (step_cnt: {step_cnt}) - Task terminating.")
                        break

            # ---------------------------------------------------------
            # 分支 B: 不开启规划 (简单 ReAct 模式)
            # ---------------------------------------------------------
            else:
                append_code_tag = AWAIT_TAG
                append_code_cnt = 0
                
                while not (
                    append_code_tag == FULFILL_INSTRUCTION
                    or append_code_cnt >= session.agent_config.execution.execution_max_number
                ):
                    small_step_cnt += 1
                    if use_moa_router:
                        result, has_physical_failure = await self._invoke_agents_concurrently_with_retry(
                            str(session.session_id), task_id, 0, small_step_cnt, 'coder', code.append_code, self, session, has_plan=False
                        )
                        if has_physical_failure:
                            observed_physical_failure = True
                        if isinstance(result, Exception):
                            logger.error(f"Coder execution (simple mode) failed: {result}")
                            append_code_tag, exec_node = (None, None) # 处理异常，将变量设置为 None
                        else:
                            append_code_tag, exec_node = self._coerce_action_node_result(
                                result,
                                "coder.append_code.simple",
                            )
                    else:
                        coder_agent = PhysicalAgent.get_physical_agent('coder')
                        agent_base_name = coder_agent.name.split('coder-', 1)[1]
                        target_zone = self._get_agent_zone('coder', agent_base_name)
                        source_zone = self.last_small_step_agent_zone
                        cells = session.chat_history.cells
                        network_latency, cell_size = self._compute_network_latency(
                            source_zone,
                            target_zone,
                            cells,
                            role="coder",
                            agent_name=agent_base_name,
                        )
                        logger.debug(f"Calculated network_latency for agent {coder_agent.name} from {source_zone} to {target_zone}: {network_latency:.4f}s, cell_size: {cell_size} bytes")
                        agent_payload = await self._invoke_agent_with_experience(
                            str(session.session_id),
                            task_id,
                            0,
                            small_step_cnt,
                            coder_agent,
                            network_latency,
                            cell_size,
                            code.append_code,
                            self,
                            session,
                            has_plan=False,
                            source_zone=source_zone,
                            target_zone=target_zone,
                            cells=cells,
                        )
                        agent_name, agent_result = self._coerce_agent_invocation_payload(
                            agent_payload,
                            "coder.append_code.simple.single",
                        )
                        append_code_tag, exec_node = self._coerce_action_node_result(
                            agent_result,
                            "coder.append_code.simple.single",
                        )
                        self.last_small_step_agent = agent_name
                        if agent_name:
                            agent_base_name = agent_name.split('coder-', 1)[1]
                            self.last_small_step_agent_zone = self._get_agent_zone('coder', agent_base_name)                    
                    await self._update_messages(session, exec_node, action_signal=append_code_tag)
                    if not exec_node or (not execution.post_code_verification(exec_node.cells_generated)):
                        if session.agent_config.debug.self_debug:
                            logger.info("$$$$$$$$$$$$$$ Error in simple mode, triggering MAS debug/filter.")
                            debug.filter_safe_cells(session, exec_node)
                            small_step_cnt += 1
                            if use_moa_router:
                                current_debug_trace = None
                                debug_payload, has_physical_failure = await self._invoke_agents_concurrently_with_retry(
                                    str(session.session_id),
                                    task_id,
                                    0,
                                    small_step_cnt,
                                    'debugger',
                                    debug.self_debug_execution,
                                    self,
                                    session,
                                    session.agent_config.debug.debug_max_number,
                                    has_plan=False,
                                    _return_selected_payload=True,
                                )
                                if has_physical_failure:
                                    observed_physical_failure = True
                                if isinstance(debug_payload, Exception):
                                    logger.error(f"Debugger execution (simple mode) failed: {debug_payload}")
                                else:
                                    _, selected_payload = (
                                        debug_payload
                                        if isinstance(debug_payload, tuple)
                                        and len(debug_payload) == 2
                                        else (debug_payload, None)
                                    )
                                    if isinstance(selected_payload, dict):
                                        current_debug_trace = selected_payload.get("debug_trace")
                                small_step_cnt += 1
                                if current_debug_trace is not None:
                                    try:
                                        filter_result, has_physical_failure = await self._invoke_agents_concurrently_with_retry(
                                            str(session.session_id),
                                            task_id,
                                            0,
                                            small_step_cnt,
                                            'filter',
                                            filter.self_debug_filter,
                                            self,
                                            session,
                                            has_plan=False,
                                            debug_trace=current_debug_trace,
                                        )
                                        if has_physical_failure:
                                            observed_physical_failure = True
                                        if isinstance(filter_result, Exception):
                                            logger.error(f"Filter execution (simple mode) failed: {filter_result}")
                                        else:
                                            filter_action_signal, filter_node = self._coerce_action_node_result(
                                                filter_result,
                                                "filter.self_debug_filter.simple.concurrent",
                                            )
                                            if filter_node is not None:
                                                await self._update_messages(
                                                    session,
                                                    filter_node,
                                                    action_signal=filter_action_signal,
                                                )
                                    finally:
                                        await self._close_debug_trace_runtime(
                                            current_debug_trace,
                                            cleanup_workspace=True,
                                        )
                                        current_debug_trace = None
                                else:
                                    filter_result, has_physical_failure = await self._invoke_agents_concurrently_with_retry(
                                        str(session.session_id),
                                        task_id,
                                        0,
                                        small_step_cnt,
                                        'filter',
                                        filter.self_debug_filter,
                                        self,
                                        session,
                                        has_plan=False,
                                    )
                                    if has_physical_failure:
                                        observed_physical_failure = True
                                    if isinstance(filter_result, Exception):
                                        logger.error(f"Filter execution (simple mode) failed: {filter_result}")
                            else:
                                debugger_agent = PhysicalAgent.get_physical_agent('debugger')
                                filter_agent = PhysicalAgent.get_physical_agent('filter')
                                current_debug_trace = create_debug_trace_record(session)

                                # Latency for debugger
                                debugger_base_name = debugger_agent.name.split('debugger-', 1)[1]
                                debugger_target_zone = self._get_agent_zone('debugger', debugger_base_name)
                                source_zone = self.last_small_step_agent_zone
                                cells = session.chat_history.cells
                                debugger_network_latency, cell_size = self._compute_network_latency(
                                    source_zone,
                                    debugger_target_zone,
                                    cells,
                                    role="debugger",
                                    agent_name=debugger_base_name,
                                )
                                logger.debug(f"Calculated network_latency for agent {debugger_agent.name} from {source_zone} to {debugger_target_zone}: {debugger_network_latency:.4f}s, cell_size: {cell_size} bytes")
                                agent_payload = await self._invoke_agent_with_experience(
                                    str(session.session_id),
                                    task_id,
                                    0,
                                    small_step_cnt,
                                    debugger_agent,
                                    debugger_network_latency,
                                    cell_size,
                                    debug.self_debug_execution,
                                    self,
                                    session,
                                    session.agent_config.debug.debug_max_number,
                                    has_plan=False,
                                    debug_trace=current_debug_trace,
                                    source_zone=source_zone,
                                    target_zone=debugger_target_zone,
                                    cells=cells,
                                )
                                agent_name, _ = self._coerce_agent_invocation_payload(
                                    agent_payload,
                                    "debug.self_debug_execution.simple.single",
                                )
                                self.last_small_step_agent = agent_name
                                if agent_name:
                                    agent_base_name = agent_name.split('debugger-', 1)[1]
                                    self.last_small_step_agent_zone = self._get_agent_zone('debugger', agent_base_name)

                                small_step_cnt += 1 # Increment for the filter agent in else block

                                # Latency for filter
                                filter_base_name = filter_agent.name.split('filter-', 1)[1]
                                filter_target_zone = self._get_agent_zone('filter', filter_base_name)
                                # The source zone might have been updated by the debugger, so we retrieve it again
                                source_zone = self.last_small_step_agent_zone
                                filter_network_latency, cell_size = self._compute_network_latency(
                                    source_zone,
                                    filter_target_zone,
                                    cells,
                                    role="filter",
                                    agent_name=filter_base_name,
                                )
                                logger.debug(f"Calculated network_latency for agent {filter_agent.name} from {source_zone} to {filter_target_zone}: {filter_network_latency:.4f}s, cell_size: {cell_size} bytes")
                                agent_payload = await self._invoke_agent_with_experience(
                                    str(session.session_id),
                                    task_id,
                                    0,
                                    small_step_cnt,
                                    filter_agent,
                                    filter_network_latency,
                                    cell_size,
                                    filter.self_debug_filter,
                                    self,
                                    session,
                                    has_plan=False,
                                    debug_trace=current_debug_trace,
                                    source_zone=source_zone,
                                    target_zone=filter_target_zone,
                                    cells=cells,
                                )
                                agent_name, filter_result = self._coerce_agent_invocation_payload(
                                    agent_payload,
                                    "filter.self_debug_filter.simple.single",
                                )
                                filter_action_signal, filter_node = self._coerce_action_node_result(
                                    filter_result,
                                    "filter.self_debug_filter.simple.single",
                                )
                                self.last_small_step_agent = agent_name
                                if agent_name:
                                    agent_base_name = agent_name.split('filter-', 1)[1]
                                    self.last_small_step_agent_zone = self._get_agent_zone('filter', agent_base_name)
                                if filter_node is not None:
                                    await self._update_messages(
                                        session,
                                        filter_node,
                                        action_signal=filter_action_signal,
                                    )
                                await self._close_debug_trace_runtime(
                                    current_debug_trace,
                                    cleanup_workspace=True,
                                )
                                current_debug_trace = None
                            # 修复后尝试继续
                            append_code_tag = AWAIT_TAG

                    append_code_cnt += 1
                    if append_code_tag == FULFILL_INSTRUCTION:
                        break

            if observed_physical_failure:
                logger.warn(f"Task {task_id} observed physical failures during concurrent agent invocation.")
            experience_collecting.update_agent_experience()
            experience_collecting.calculate_and_log_task_summary(
                str(session.session_id),
                task_id,
                router_strategy=self.agent_llm_manager.selection_strategy,
            )
            await self.save_session_content(session)
            assert isinstance(session.chat_history, CellHistoryMemory)
            return session.chat_history.fetch_response()



    def _get_session_content(self, session: Session):

        return SessionContent(
            llm_config=self.llm_config,
            agent_config=session.agent_config,
            code_executor_config=session.code_executor_config,
            user_info=UserInfo(user_id=self.user_id, user_name=self.user_name),
            session_info=SessionInfo(
                session_id=session.session_id, session_name=session.session_name
            ),
            chat_history=session.chat_history,
            fs_session_root_dir=session.workspace.root_dir,
        )

    async def _broadcast_session_update(self, session: Session):
        session_content = self._get_session_content(session)

        await self.agent_manager.broadcast_session_update(
            self.user_id, session.session_id, session_content
        )

    async def _update_messages(
        self, session: Session, node: Node, action_signal: str = AWAIT_TAG
    ):
        if node is None:
            logger.warn("Attempted to update messages with a None node. Skipping.")
            return

        chat_history = session.chat_history
        chat_history.add_messages(node, action_signal)

        if self.agent_manager != None:
            await self._broadcast_session_update(session)

    async def save_session_content(self, session: Session):

        session_content = self._get_session_content(session)
        user_id = session_content.user_info.user_id
        session_id = session_content.session_info.session_id
        session_file_path = session.session_json_path
        session_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if the session file already exists
        is_update = os.path.exists(session_file_path)
        # 异步写入文件
        async with aiofiles.open(session_file_path, "w") as f:
            await f.write(session_content.model_dump_json())

        if is_update:
            logger.info(
                f"Session content has been UPDATED successfully for user id: {user_id}, session id: {session_id}!"
            )
        else:
            logger.info(
                f"Session content has been SAVED successfully for user id: {user_id}, session id: {session_id}!"
            )

    def _extract_before_cell_content(
        self, content: str | list | None, cell_content: str
    ) -> str:
        content = content_str(content)
        position = content.find(cell_content)
        if position != -1:
            return content[:position]
        else:
            return ""

    def _extract_after_cell_content(
        self, content: str | list | None, cell_content: str
    ) -> str:
        content = content_str(content)
        position = content.find(cell_content)
        if position != -1:
            return content[position + len(cell_content) :]
        else:
            return ""

    async def _self_debug(
        self,
        session: Session,
        current_node: ExecutionNode | StepNode,
        # init_generated_cells: List[Union[MarkdownCell, CodeCell]],
        # init_output_cells: List[CodeOutputCell],
        self_debug_max_number: int = 8,
        has_plan: bool = False,
    ) -> str:
        # Filter the safe cells in the beginning to update the chat history
        cells_to_debug = debug.filter_safe_cells(session, current_node)

        # EXECUTION in a loop
        await debug.self_debug_execution(
            self, session, self_debug_max_number, has_plan=has_plan
        )

        # FILTER
        debug_tag = await filter.self_debug_filter(self, session, has_plan=has_plan)
        # DEBUG_FAIL_TAG | DEBUG_SUCCEED_TAG

        return debug_tag
