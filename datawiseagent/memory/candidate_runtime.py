from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datawiseagent.coding import JupyterCodeExecutor
from datawiseagent.coding.base import CodeBlock, IPythonCodeResult
from datawiseagent.coding.jupyter.base import JupyterConnectionInfo
from datawiseagent.common.log import logger
from datawiseagent.common.types import CodeCell, CodeOutputCell, MarkdownCell, NotebookCell
from datawiseagent.memory.chat_history import CellHistoryMemory
from datawiseagent.memory.session import Session

MAX_KERNEL_RECOVERY_ATTEMPTS = 2


def clone_chat_history(chat_history: CellHistoryMemory) -> CellHistoryMemory:
    return CellHistoryMemory.from_json(chat_history.model_dump(mode="json"))


@dataclass
class CandidateWorkspace:
    candidate_id: str
    root_dir: Path
    input_dir: Path
    system_dir: Path
    working_dir: Path
    display_dir: Path
    input_target: Path
    system_target: Path


def _link_or_copy_dir(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    relative_source = os.path.relpath(source, target.parent)
    try:
        target.symlink_to(relative_source, target_is_directory=True)
    except OSError as exc:
        logger.warn(
            f"[CandidateWorkspace] Failed to symlink '{target}' -> '{relative_source}', "
            f"falling back to copytree: {exc}"
        )
        shutil.copytree(source, target, dirs_exist_ok=True)


def create_candidate_workspace(base_session: Session, candidate_id: str) -> CandidateWorkspace:
    root_dir = base_session.root_dir / ".branches" / candidate_id
    if root_dir.exists():
        shutil.rmtree(root_dir, ignore_errors=True)

    root_dir.mkdir(parents=True, exist_ok=True)

    input_dir = root_dir / "input"
    system_dir = root_dir / "system"
    working_dir = root_dir / "working"
    display_dir = root_dir / "display"

    _link_or_copy_dir(base_session.input_dir, input_dir)
    _link_or_copy_dir(base_session.system_dir, system_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    display_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[CandidateWorkspace] initialized candidate_id={candidate_id} root={root_dir}"
    )

    return CandidateWorkspace(
        candidate_id=candidate_id,
        root_dir=root_dir,
        input_dir=input_dir,
        system_dir=system_dir,
        working_dir=working_dir,
        display_dir=display_dir,
        input_target=base_session.input_dir,
        system_target=base_session.system_dir,
    )


def cleanup_candidate_workspace(candidate_workspace: CandidateWorkspace) -> None:
    shutil.rmtree(candidate_workspace.root_dir, ignore_errors=True)


class CandidateFileSystem:
    def __init__(self, candidate_workspace: CandidateWorkspace):
        self.root_dir = candidate_workspace.root_dir
        self.input_dir = candidate_workspace.input_dir
        self.system_dir = candidate_workspace.system_dir
        self.working_dir = candidate_workspace.working_dir
        self.display_dir = candidate_workspace.display_dir
        self._input_target = candidate_workspace.input_target
        self._system_target = candidate_workspace.system_target

    def _ensure_shared_dirs(self) -> None:
        if not self.input_dir.exists():
            _link_or_copy_dir(self._input_target, self.input_dir)
        if not self.system_dir.exists():
            _link_or_copy_dir(self._system_target, self.system_dir)

    def initialize(self, maintained_dirs: list[str | Path]) -> None:
        self._ensure_shared_dirs()

        resolved_maintained_dirs = {
            Path(one_dir).resolve()
            for one_dir in maintained_dirs
            if Path(one_dir).exists()
        }

        for item in self.root_dir.iterdir():
            if item.resolve() in resolved_maintained_dirs:
                continue

            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as exc:
                logger.warn(f"[CandidateWorkspace] Failed to remove {item}: {exc}")

        self._ensure_shared_dirs()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.display_dir.mkdir(parents=True, exist_ok=True)

    def fetch_workspace_status(self) -> str:
        result = subprocess.run(
            ["tree", "-Fsh", "-I", "__pycache__|*.pyc", "./"],
            capture_output=True,
            text=True,
            cwd=self.root_dir,
        )

        if result.returncode == 0:
            return result.stdout
        return "Error fetching workspace status: " + result.stderr


class CandidateSessionView:
    is_candidate_runtime = True

    def __init__(
        self,
        base_session: Session,
        candidate_workspace: CandidateWorkspace,
        chat_history: CellHistoryMemory,
        workspace: CandidateFileSystem,
        code_executor: JupyterCodeExecutor,
        workspace_root_dir: str,
    ):
        self._base_session = base_session
        self._candidate_workspace = candidate_workspace
        self._workspace_root_dir = workspace_root_dir
        self._workspace_bootstrapped = False
        self._closed = False

        self.session_id = base_session.session_id
        self.session_name = base_session.session_name
        self.candidate_id = candidate_workspace.candidate_id

        self.chat_history = chat_history
        self.root_dir = candidate_workspace.root_dir
        self.display_dir = candidate_workspace.display_dir
        self.input_dir = candidate_workspace.input_dir
        self.system_dir = candidate_workspace.system_dir
        self.working_dir = candidate_workspace.working_dir

        self.workspace = workspace
        self.code_executor = code_executor

        self.agent_config = base_session.agent_config
        self.code_executor_config = base_session.code_executor_config
        self.current_user_query = base_session.current_user_query
        self.last_selected_result = base_session.last_selected_result
        self.last_selected_step_id = base_session.last_selected_step_id
        self.last_activity_time = base_session.last_activity_time

    @classmethod
    async def create(
        cls,
        base_session: Session,
        candidate_id: str,
        source_chat_history: Optional[CellHistoryMemory] = None,
    ) -> "CandidateSessionView":
        assert isinstance(base_session.chat_history, CellHistoryMemory)
        assert isinstance(base_session.code_executor, JupyterCodeExecutor)

        candidate_workspace = create_candidate_workspace(base_session, candidate_id)
        workspace = CandidateFileSystem(candidate_workspace)
        code_executor: Optional[JupyterCodeExecutor] = None

        try:
            connection_info = cls._extract_connection_info(base_session.code_executor)
            workspace_root_dir = cls._build_workspace_root_dir(
                base_session, candidate_workspace
            )
            kernel_name = getattr(base_session.code_executor, "_kernel_name", "python3")
            timeout = getattr(base_session.code_executor, "_timeout", 600)
            code_executor = await JupyterCodeExecutor.create(
                jupyter_server=connection_info,
                kernel_name=kernel_name,
                timeout=timeout,
                output_dir=candidate_workspace.display_dir,
                use_docker_space=base_session.code_executor.use_docker_space,
            )

            session_view = cls(
                base_session=base_session,
                candidate_workspace=candidate_workspace,
                chat_history=clone_chat_history(
                    source_chat_history or base_session.chat_history
                ),
                workspace=workspace,
                code_executor=code_executor,
                workspace_root_dir=workspace_root_dir,
            )
            await session_view._rebuild_runtime_from_chat_history(restart_kernel=False)
            return session_view
        except Exception:
            if code_executor is not None:
                try:
                    await code_executor.stop()
                except Exception as exc:
                    logger.warn(
                        f"Failed to stop candidate executor during create cleanup "
                        f"for '{candidate_id}': {exc}"
                    )

            cleanup_candidate_workspace(candidate_workspace)
            logger.info(
                f"[CandidateWorkspace] cleaned candidate_id={candidate_id} "
                f"root={candidate_workspace.root_dir} during create failure"
            )
            raise

    @staticmethod
    def _extract_connection_info(
        code_executor: JupyterCodeExecutor,
    ) -> JupyterConnectionInfo:
        jupyter_server = code_executor.jupyter_server
        if isinstance(jupyter_server, JupyterConnectionInfo):
            return jupyter_server
        return jupyter_server.connection_info

    @staticmethod
    def _build_workspace_root_dir(
        base_session: Session,
        candidate_workspace: CandidateWorkspace,
    ) -> str:
        workspace_root_dir = Path(str(base_session.workspace_root_dir()))
        relative_root = candidate_workspace.root_dir.relative_to(base_session.root_dir)
        return str(workspace_root_dir / relative_root)

    async def _ensure_workspace_context(self) -> None:
        if self._workspace_bootstrapped:
            return

        bootstrap_code = (
            "import os\n"
            f"os.makedirs(r'''{self._workspace_root_dir}''', exist_ok=True)\n"
            f"os.chdir(r'''{self._workspace_root_dir}''')\n"
        )
        await self.code_executor.execute_code_blocks(
            CodeBlock(code=bootstrap_code, language="python")
        )
        self._workspace_bootstrapped = True

    async def _rebuild_runtime_from_chat_history(self, restart_kernel: bool) -> None:
        if restart_kernel:
            await self.code_executor.restart()

        self.workspace.initialize([self.input_dir, self.system_dir])
        self._workspace_bootstrapped = False
        await self._ensure_workspace_context()

        new_cells: list[NotebookCell] = []
        code_cells_stack: list[CodeCell] = []
        for cell in self.chat_history.cells:
            if isinstance(cell, MarkdownCell):
                new_cells.append(cell)
            elif isinstance(cell, CodeCell):
                code_cells_stack.append(cell)

                if cell.code_output is not None:
                    cell.code_output.code_result = None

                new_cells.append(cell)
            elif isinstance(cell, CodeOutputCell):
                if len(code_cells_stack) != 0:
                    for code_cell in code_cells_stack:
                        if (
                            code_cell.code_output is not None
                            and "ERROR: Timeout waiting for output from code block."
                            in code_cell.code_output.to_string()
                        ):
                            code_output = await self.code_executor.execute_code_blocks(
                                code_cell.to_code_block(), custom_timeout=0.1
                            )
                        else:
                            code_output = await self.code_executor.execute_code_blocks(
                                code_cell.to_code_block()
                            )

                        if code_cell.code_output is not None:
                            code_cell.code_output.update_code_result(code_output)
                        else:
                            code_cell.code_output = CodeOutputCell(
                                code_result=code_output
                            )

                        new_cells.append(code_cell.code_output)
                    code_cells_stack.clear()

        self.chat_history.cells = new_cells

    async def rerun_cells(self) -> None:
        await self._rebuild_runtime_from_chat_history(restart_kernel=True)

    async def safe_execute_code_blocks(
        self, code_blocks: CodeBlock | list[CodeBlock]
    ) -> IPythonCodeResult:
        await self._ensure_workspace_context()

        if isinstance(self.code_executor, JupyterCodeExecutor):
            last_error: Optional[RuntimeError] = None
            for attempt in range(MAX_KERNEL_RECOVERY_ATTEMPTS + 1):
                try:
                    await self.code_executor.check_jupyter_kernel_health()
                    return await self.code_executor.execute_code_blocks(code_blocks)
                except RuntimeError as exc:
                    last_error = exc
                    if attempt >= MAX_KERNEL_RECOVERY_ATTEMPTS:
                        break
                    await self.rerun_cells()

            raise RuntimeError(
                f"Kernel recovery failed after {MAX_KERNEL_RECOVERY_ATTEMPTS + 1} attempts."
            ) from last_error

        return await self.code_executor.execute_code_blocks(code_blocks)

    def workspace_root_dir(self) -> str:
        return self._workspace_root_dir

    async def close(
        self,
        cleanup_workspace: bool = False,
        stop_executor: bool = True,
    ) -> None:
        if self._closed:
            return

        if stop_executor and self.code_executor is not None:
            try:
                await self.code_executor.stop()
            except Exception as exc:
                logger.warn(
                    f"Failed to stop candidate executor '{self.candidate_id}': {exc}"
                )

        if cleanup_workspace:
            cleanup_candidate_workspace(self._candidate_workspace)
            logger.info(
                f"[CandidateWorkspace] cleaned candidate_id={self.candidate_id} "
                f"root={self.root_dir}"
            )

        self._closed = True
