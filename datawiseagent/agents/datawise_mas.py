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
from typing import List, Optional, Literal, Union, Tuple, TYPE_CHECKING
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
import datawiseagent.agents.utils.execution as execution
import datawiseagent.agents.utils.parser as parser
import datawiseagent.agents.utils.prompt_tool as prompt_tool
import time
from datawiseagent.agents.os.experience import experience_collecting

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


class DatawiseMAS:
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
        Initialize a `DatawiseAgent` bound to a single user's workspace.

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
!pip install -qqq numpy pandas matplotlib scipy scikit-learn torch

# Get system information (CPU, GPU, Memory)
!echo "CPU Cores: $(lscpu | grep '^CPU(s):' | awk '{print $2}')"
!echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits)"
!echo "Memory Usage: $(free -h | grep Mem | awk '{print $2 \" total, \" $3 \" used, \" $4 \" free\"}')"

import pandas as pd
# Disable HTML representation of dataframe
pd.options.display.notebook_repr_html = False
""",
                role="assistant",
                name="Datawise_Agent",
            )

            c2 = CodeCell(
                content="# You could use sklearn or pytorch for data modeling.\n!pip show numpy pandas matplotlib scipy scikit-learn torch| grep Version",
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
                content="!grep -E '^(NAME|VERSION|ID|ID_LIKE)=' /etc/os-release\n!pip install -qqq numpy pandas matplotlib scipy scikit-learn\n\nimport pandas as pd\n# disable HTML representation of dataframe\npd.options.display.notebook_repr_html = False",
                role="assistant",
                name="Datawise_Agent",
            )

            c2 = CodeCell(
                content="!pip show numpy pandas matplotlib scipy scikit-learn | grep Version",
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
    # 物理移植，分成4个agent
    # async def chat(
    #     self,
    #     session_id: uuid.UUID,
    #     query: str,
    #     agent_config: dict = agent_config,
    #     work_mode: Literal["jupyter", "jupyter+script"] = "jupyter",
    # ):
    #     """
    #     Handle a user query: generate cells, execute, (optionally) plan, and (optionally) self-debug.

    #     This is the main entry point for chat turns. It updates the session's chat
    #     history with user/assistant cells and executes code incrementally.

    #     Parameters
    #     ----------
    #     session_id : uuid.UUID
    #         Target session.
    #     query : str
    #         User query or instruction.
    #     agent_config : dict, default=agent_config
    #         Serialized `DatawiseAgentConfig` dict controlling planning/execution/debug limits.
    #     work_mode : {"jupyter","jupyter+script"}, default="jupyter"
    #         If "jupyter+script", appends a hybrid-workflow hint to the query.

    #     Returns
    #     -------
    #     list[NotebookCell]
    #         The latest assistant-visible cells (as returned by `chat_history.fetch_response()`).

    #     Raises
    #     ------
    #     KeyError
    #         If the session does not exist.
    #     AssertionError
    #         If required components (e.g., code executor) are missing.
    #     """

    #     datawise_agent_config = DatawiseAgentConfig(**agent_config)
    #     session: Session = self.sessions[session_id]
    #     session.agent_config = datawise_agent_config

    #     if work_mode == "jupyter":
    #         user_msg = LLMResult(role="user", name="USER", content=query)

    #     elif work_mode == "jupyter+script":
    #         query = query + HYBRID_WORKFLOW
    #         user_msg = LLMResult(role="user", name="USER", content=query)

    #     user_cell = NotebookCell.llm_result_convert(
    #         user_msg, parse_mode=ConvertType.CONVERT_USER_CELL
    #     )
    #     user_node = UserNode(cells_generated=user_cell)

    #     session.current_user_query = query

    #     # session.chat_history.add_messages(user_node)
    #     await self._update_messages(session, user_node)

    #     # if session.agent_config.plan.planning:

    #     #     step_cnt = 0

    #     #     # initial plan -> append code
    #     #     debug_signal = await planner.initiate_step(
    #     #         self,
    #     #         session,
    #     #         session.agent_config.debug.self_debug,
    #     #         session.agent_config.debug.debug_max_number,
    #     #     )
    #     #     step_cnt += 1

    #     #     class State(Enum):
    #     #         INITIATE_PLAN = "initiate_plan"
    #     #         APPEND_CODE = "append_code"
    #     #         UPDATE_PLAN = "update_plan"
    #     #         TERMINATE = "terminate"

    #     #     state = State.INITIATE_PLAN
    #     #     action_signal: str = None
    #     #     debug_signal: str = None
    #     #     planning_cnt: int = 1
    #     #     append_code_cnt = 0

    #     #     max_debug_capacity = 100
    #     #     debug_cnt = 0
    #     #     debug_failure_per_step = 0
    #     #     while not (planning_cnt >= session.agent_config.plan.planning_max_number):

    #     #         step_cnt += 1
    #     #         if debug_signal != None:
    #     #             debug_cnt += 1
    #     #             if debug_signal == DEBUG_FAIL_TAG:
    #     #                 debug_failure_per_step += 1
    #     #             debug_signal = None

    #     #         logger.debug(f"debug_failure_per_step : {debug_failure_per_step}")

    #     #         is_self_debug = (
    #     #             session.agent_config.debug.self_debug
    #     #             and debug_cnt < max_debug_capacity
    #     #             and (
    #     #                 (
    #     #                     session.agent_config.max_debug_by_step != None
    #     #                     and debug_failure_per_step
    #     #                     <= session.agent_config.max_debug_by_step
    #     #                 )
    #     #                 or session.agent_config.max_debug_by_step == None
    #     #             )
    #     #         )

    #     #         if state == State.INITIATE_PLAN:
    #     #             append_code_cnt = 0
    #     #             debug_failure_per_step = 0
    #     #             action_signal, debug_signal = await code.append_code(
    #     #                 self,
    #     #                 session,
    #     #                 is_self_debug,
    #     #                 session.agent_config.debug.debug_max_number,
    #     #                 has_plan=True,
    #     #             )
    #     #             state = State.APPEND_CODE
    #     #         # if state == State.INITIATE_PLAN:
    #     #         #     # 调用修改后的 planner，只拿结果
    #     #         #     action_signal, step_node = await planner.initiate_step(self, session)
                    
    #     #         #     # 控制器亲自检查：代码跑通了吗？
    #     #         #     # 调用 utils 里的验证函数
    #     #         #     is_success = execution.post_code_verification(step_node.cells_generated)
                    
    #     #         #     if not is_success:
    #     #         #         logger.warn("Planner output has errors. Transitioning to DEBUG state.")
    #     #         #         # 记录当前出错的节点，传给下一步的 Debugger
    #     #         #         self.current_failed_node = step_node 
    #     #         #         state = State.DEBUG
    #     #         #     else:
    #     #         #         logger.info("Planner output is clean. Transitioning to APPEND_CODE.")
    #     #         #         state = State.APPEND_CODE

    #     #         elif state == State.APPEND_CODE:
    #     #             append_code_cnt += 1
    #     #             if (
    #     #                 action_signal == AWAIT_TAG
    #     #                 and append_code_cnt
    #     #                 < session.agent_config.execution.execution_max_number
    #     #             ):
    #     #                 action_signal, debug_signal = await code.append_code(
    #     #                     self,
    #     #                     session,
    #     #                     is_self_debug,
    #     #                     session.agent_config.debug.debug_max_number,
    #     #                     has_plan=True,
    #     #                 )
    #     #                 state = State.APPEND_CODE

    #     #             else:
    #     #                 # assert (action_signal == END_STEP_TAG) or (action_signal == AWAIT_TAG and append_code_cnt > execution_max_number)
    #     #                 action_signal, debug_signal = await planner.update_plan(
    #     #                     self,
    #     #                     session,
    #     #                     is_self_debug,
    #     #                     session.agent_config.debug.debug_max_number,
    #     #                 )
    #     #                 state = State.UPDATE_PLAN
    #     #         elif state == State.UPDATE_PLAN:
    #     #             append_code_cnt = 0
    #     #             debug_failure_per_step = 0
    #     #             if (
    #     #                 action_signal == ITERATE_ON_LAST_STEP
    #     #                 or action_signal == ADVANCE_TO_NEXT_STEP
    #     #             ):
    #     #                 planning_cnt += 1
    #     #                 action_signal, debug_signal = await code.append_code(
    #     #                     self,
    #     #                     session,
    #     #                     is_self_debug,
    #     #                     session.agent_config.debug.debug_max_number,
    #     #                     has_plan=True,
    #     #                 )
    #     #                 state = State.APPEND_CODE
    #     #             elif action_signal == FULFILL_INSTRUCTION:
    #     #                 state = State.TERMINATE
    #     #                 debug_signal = None
    #     #         elif state == State.TERMINATE:
    #     #             debug_signal = None
    #     #             break

    #     #         logger.info(
    #     #             f"Current STEP CNT: {step_cnt} with MAX STEP NUMBER {session.agent_config.max_step_number}"
    #     #         )

    #     #         if (
    #     #             session.agent_config.max_step_number != None
    #     #             and step_cnt >= session.agent_config.max_step_number
    #     #         ):
    #     #             debug_signal = None
    #     #             break
    #     if session.agent_config.plan.planning:
    #         # 状态枚举
    #         class State(Enum):
    #             INITIATE_PLAN = "initiate_plan"
    #             APPEND_CODE = "append_code"
    #             DEBUG = "debug"
    #             FILTER = "filter"
    #             UPDATE_PLAN = "update_plan"
    #             TERMINATE = "terminate"

    #         step_cnt = 0
    #         planning_cnt = 1  # 原逻辑初次 plan 后设为 1
    #         append_code_cnt = 0
    #         debug_cnt = 0
    #         debug_failure_per_step = 0
    #         max_debug_capacity = 100
            
    #         state = State.INITIATE_PLAN
    #         action_signal = None
    #         current_failed_node = None

    #         # 外层循环条件对齐原逻辑
    #         while not (planning_cnt >= session.agent_config.plan.planning_max_number):
                
    #             step_cnt += 1
    #             logger.info(f"Current STEP CNT: {step_cnt} with MAX STEP NUMBER {session.agent_config.max_step_number}")
                
    #             # --- 安全熔断检查 ---
    #             if (session.agent_config.max_step_number is not None 
    #                 and step_cnt >= session.agent_config.max_step_number):
    #                 break

    #             # --- 核心：判定当前是否允许自调试 (完全对齐原逻辑计算方式) ---
    #             is_self_debug = (
    #                 session.agent_config.debug.self_debug
    #                 and debug_cnt < max_debug_capacity
    #                 and (
    #                     (
    #                         session.agent_config.max_debug_by_step is not None
    #                         and debug_failure_per_step <= session.agent_config.max_debug_by_step
    #                     )
    #                     or session.agent_config.max_debug_by_step is None
    #                 )
    #             )

    #             if state == State.INITIATE_PLAN:
    #                 # 初始规划
    #                 append_code_cnt = 0
    #                 debug_failure_per_step = 0
    #                 action_signal, step_node = await planner.initiate_step(self, session)
                    
    #                 if not execution.post_code_verification(step_node.cells_generated) and is_self_debug:
    #                     current_failed_node = step_node
    #                     state = State.DEBUG
    #                 else:
    #                     state = State.APPEND_CODE

    #             elif state == State.APPEND_CODE:
    #                 append_code_cnt += 1
                    
    #                 # 检查是否超过当前步骤的最大执行次数 (对齐原逻辑 if action_signal == AWAIT_TAG and ...)
    #                 if append_code_cnt > session.agent_config.execution.execution_max_number:
    #                     state = State.UPDATE_PLAN
    #                     continue

    #                 action_signal, exec_node = await code.append_code(self, session, has_plan=True)

    #                 if not execution.post_code_verification(exec_node.cells_generated) and is_self_debug:
    #                     current_failed_node = exec_node
    #                     state = State.DEBUG
    #                 else:
    #                     # 如果没有报错，且 action_signal 是 AWAIT，继续 APPEND_CODE 循环
    #                     if action_signal == AWAIT_TAG:
    #                         state = State.APPEND_CODE
    #                     else:
    #                         state = State.UPDATE_PLAN

    #             elif state == State.DEBUG:
    #                 # 调试计数器增加 (对齐原逻辑 debug_cnt += 1, debug_failure_per_step += 1)
    #                 debug_cnt += 1
    #                 debug_failure_per_step += 1
                    
    #                 # 执行调试
    #                 debug.filter_safe_cells(session, current_failed_node)
    #                 await debug.self_debug_execution(
    #                     self, session, session.agent_config.debug.debug_max_number, has_plan=True
    #                 )
    #                 state = State.FILTER

    #             elif state == State.FILTER:
    #                 # 执行过滤
    #                 debug_tag = await filter.self_debug_filter(self, session, has_plan=True)
                    
    #                 # 这里的逻辑判定对齐原代码：
    #                 # 如果过滤失败（DEBUG_FAIL_TAG），在 APPEND 模式下会设 action_signal = AWAIT 继续尝试
    #                 if debug_tag == DEBUG_FAIL_TAG:
    #                     state = State.APPEND_CODE
    #                 else:
    #                     # 调试成功或过滤结束，返回 APPEND 状态继续
    #                     state = State.APPEND_CODE

    #             elif state == State.UPDATE_PLAN:
    #                 # 重置步骤内计数器 (对齐原逻辑进入下一步/迭代前重置)
    #                 append_code_cnt = 0
    #                 debug_failure_per_step = 0
                    
    #                 action_signal, step_node = await planner.update_plan(self, session)

    #                 if action_signal in [ITERATE_ON_LAST_STEP, ADVANCE_TO_NEXT_STEP]:
    #                     planning_cnt += 1 # 只有明确推进或迭代才增加 planning_cnt
                        
    #                     # 检查新计划带的代码是否有错
    #                     if not execution.post_code_verification(step_node.cells_generated) and is_self_debug:
    #                         current_failed_node = step_node
    #                         state = State.DEBUG
    #                     else:
    #                         state = State.APPEND_CODE
    #                 elif action_signal == FULFILL_INSTRUCTION:
    #                     state = State.TERMINATE
    #                 else:
    #                     # 其他情况默认终止
    #                     state = State.TERMINATE

    #             elif state == State.TERMINATE:
    #                 break

    #     else:
    #         # WITHOUT PLANNING
    #         # It's more like ReAct agent with self-debug mechanism, and the working style is to incrementally generate and execute code snippnets and the code snippet could reference previous variables and intermediate results.
    #         append_code_tag = AWAIT_TAG
    #         append_code_cnt = 0
    #         while not (
    #             append_code_tag == FULFILL_INSTRUCTION
    #             or append_code_cnt
    #             >= session.agent_config.execution.execution_max_number
    #         ):

    #             is_self_debug = (session.agent_config.debug.self_debug,)
    #             append_code_tag, debug_tag = await code.append_code(
    #                 self, session, is_self_debug, session.agent_config.debug.debug_max_number
    #             )

    #             append_code_cnt += 1
    #             # When only append code and self debug
    #             if debug_tag == DEBUG_FAIL_TAG:
    #                 # If it failed in debugging phase, continue to append code.
    #                 append_code_tag = AWAIT_TAG

    #             if append_code_tag == FULFILL_INSTRUCTION:
    #                 # TODO: evaluation or summary
    #                 if session.agent_config.evaluation:
    #                     pass
    #     await self.save_session_content(session)

    #     assert isinstance(session.chat_history, CellHistoryMemory)

    #     return session.chat_history.fetch_response()


    async def _invoke_agent_with_experience(self, session_id: str, task_id: str, physical_agent: PhysicalAgent, agent_function, *args, **kwargs):
        """一个包装器，用于调用子代理函数，并使用其唯一名称记录性能数据。"""
        start_time = time.time()
        result = await agent_function(physical_agent, *args, **kwargs)
        end_time = time.time()
        latency = end_time - start_time

        token_count = 0
        if result and isinstance(result, tuple) and len(result) > 1:
            node = result[1]
            if isinstance(node, Node) and hasattr(node, 'completion_usage') and node.completion_usage and hasattr(node.completion_usage, 'total_tokens'):
                token_count = node.completion_usage.total_tokens
        logger.debug(f"Extracted token_count for {physical_agent.name} (task_id: {task_id}): {token_count}")

        experience_collecting.log_task_experience(session_id, task_id, physical_agent.name, latency, token_count)

        return result

    async def chat(
            self,
            session_id: uuid.UUID,
            query: str,
            agent_config: dict = agent_config,
            work_mode: Literal["jupyter", "jupyter+script"] = "jupyter",
        ):
            DatawiseMAS._global_task_counter += 1
            task_id = str(DatawiseMAS._global_task_counter) # Use sequential task_id
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
                        append_code_cnt = 0
                        debug_failure_per_step = 0
                        planner_agent = PhysicalAgent.get_physical_agent('planner')
                        action_signal, step_node = await self._invoke_agent_with_experience(
                            str(session.session_id),
                            task_id, # Pass task_id
                            planner_agent,
                            planner.initiate_step,
                            self, session
                        )
                        
                        if not execution.post_code_verification(step_node.cells_generated) and is_self_debug:
                            current_failed_node = step_node
                            state = State.DEBUG
                        else:
                            state = State.APPEND_CODE

                    elif state == State.APPEND_CODE:
                        append_code_cnt += 1
                        if append_code_cnt > session.agent_config.execution.execution_max_number:
                            state = State.UPDATE_PLAN
                            continue

                        coder_agent = PhysicalAgent.get_physical_agent('coder')
                        action_signal, exec_node = await self._invoke_agent_with_experience(
                            str(session.session_id),
                            task_id, # Pass task_id
                            coder_agent,
                            code.append_code,
                            self, session, has_plan=True
                        )

                        if not execution.post_code_verification(exec_node.cells_generated) and is_self_debug:
                            current_failed_node = exec_node
                            state = State.DEBUG
                        else:
                            if action_signal == AWAIT_TAG:
                                state = State.APPEND_CODE
                            else:
                                state = State.UPDATE_PLAN

                    elif state == State.DEBUG:
                        debug_cnt += 1
                        debug_failure_per_step += 1
                        debugger_agent = PhysicalAgent.get_physical_agent('debugger')
                        debug.filter_safe_cells(session, current_failed_node)
                        await self._invoke_agent_with_experience(
                            str(session.session_id),
                            task_id, # Pass task_id
                            debugger_agent,
                            debug.self_debug_execution,
                            self, session, session.agent_config.debug.debug_max_number, has_plan=True
                        )
                        state = State.FILTER

                    elif state == State.FILTER:
                        filter_agent = PhysicalAgent.get_physical_agent('filter')
                        filter_result = await self._invoke_agent_with_experience(
                            str(session.session_id),
                            task_id, # Pass task_id
                            filter_agent,
                            filter.self_debug_filter,
                            self, session, has_plan=True
                        )
                        debug_tag = filter_result[0]
                        # 无论修复结果如何，都回到 APPEND_CODE 让程序员继续尝试或由主循环推进
                        state = State.APPEND_CODE

                    elif state == State.UPDATE_PLAN:
                        append_code_cnt = 0
                        debug_failure_per_step = 0
                        planner_agent = PhysicalAgent.get_physical_agent('planner')
                        action_signal, step_node = await self._invoke_agent_with_experience(
                            str(session.session_id),
                            task_id, # Pass task_id
                            planner_agent,
                            planner.update_plan,
                            self, session
                        )

                        if action_signal in [ITERATE_ON_LAST_STEP, ADVANCE_TO_NEXT_STEP]:
                            planning_cnt += 1
                            if not execution.post_code_verification(step_node.cells_generated) and is_self_debug:
                                current_failed_node = step_node
                                state = State.DEBUG
                            else:
                                state = State.APPEND_CODE
                        elif action_signal == FULFILL_INSTRUCTION:
                            state = State.TERMINATE
                        else:
                            state = State.TERMINATE

                    elif state == State.TERMINATE:
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
                    coder_agent = PhysicalAgent.get_physical_agent('coder')
                    append_code_tag, exec_node = await self._invoke_agent_with_experience(
                        str(session.session_id),
                        task_id, # Pass task_id
                        coder_agent,
                        code.append_code,
                        self, session, has_plan=False
                    )
                    
                    if not execution.post_code_verification(exec_node.cells_generated):
                        if session.agent_config.debug.self_debug:
                            logger.info("Error in simple mode, triggering MAS debug/filter.")
                            debugger_agent = PhysicalAgent.get_physical_agent('debugger')
                            filter_agent = PhysicalAgent.get_physical_agent('filter')
                            debug.filter_safe_cells(session, exec_node)
                            await self._invoke_agent_with_experience(
                                str(session.session_id),
                                task_id, # Pass task_id
                                debugger_agent,
                                debug.self_debug_execution,
                                self, session, session.agent_config.debug.debug_max_number, has_plan=False
                            )
                            await self._invoke_agent_with_experience(
                                str(session.session_id),
                                task_id, # Pass task_id
                                filter_agent,
                                filter.self_debug_filter,
                                self, session, has_plan=False
                            )
                            # 修复后尝试继续
                            append_code_tag = AWAIT_TAG

                    append_code_cnt += 1
                    if append_code_tag == FULFILL_INSTRUCTION:
                        break

            experience_collecting.update_agent_experience()
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




