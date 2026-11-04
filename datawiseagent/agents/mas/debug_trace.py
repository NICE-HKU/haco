from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from datawiseagent.coding import JupyterCodeExecutor
from datawiseagent.coding.base import CodeBlock, IPythonCodeResult
from datawiseagent.common.types import CodeCell, CodeOutputCell, MarkdownCell, NotebookCell
from datawiseagent.common.types.node import DebugNode, PostDebuggingNode
from datawiseagent.memory.chat_history import CellHistoryMemory
from datawiseagent.memory.session import Session

MAX_KERNEL_RECOVERY_ATTEMPTS = 2


def clone_chat_history(chat_history: CellHistoryMemory) -> CellHistoryMemory:
    return CellHistoryMemory.from_json(chat_history.model_dump(mode="json"))


class TempDebugSessionView:
    """A lightweight session view that keeps a private chat history for debug/filter."""

    def __init__(self, base_session: Any, chat_history: CellHistoryMemory):
        self._base_session = base_session
        self.session_id = base_session.session_id
        self.session_name = base_session.session_name

        self.chat_history = chat_history
        self.root_dir = base_session.root_dir
        self.display_dir = base_session.display_dir
        self.input_dir = base_session.input_dir
        self.system_dir = base_session.system_dir
        self.working_dir = base_session.working_dir

        self.workspace = base_session.workspace
        self.code_executor = base_session.code_executor

        self.agent_config = base_session.agent_config
        self.code_executor_config = base_session.code_executor_config
        self.current_user_query = base_session.current_user_query
        self.last_selected_result = base_session.last_selected_result
        self.last_selected_step_id = base_session.last_selected_step_id
        self.last_activity_time = base_session.last_activity_time

    async def rerun_cells(self):
        await self.code_executor.restart()

        self.workspace.initialize([self.input_dir, self.system_dir])

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

    async def safe_execute_code_blocks(
        self, code_blocks: CodeBlock | list[CodeBlock]
    ) -> IPythonCodeResult:
        if isinstance(self.code_executor, JupyterCodeExecutor):
            last_error: Optional[RuntimeError] = None
            for attempt in range(MAX_KERNEL_RECOVERY_ATTEMPTS + 1):
                try:
                    await self.code_executor.check_jupyter_kernel_health()
                    return await self.code_executor.execute_code_blocks(code_blocks)
                except RuntimeError as e:
                    if not self._is_recoverable_jupyter_error(e):
                        raise
                    last_error = e

                if attempt >= MAX_KERNEL_RECOVERY_ATTEMPTS:
                    break

                try:
                    await self.rerun_cells()
                except RuntimeError as rerun_error:
                    if not self._is_recoverable_jupyter_error(rerun_error):
                        raise
                    last_error = rerun_error

            raise RuntimeError(
                f"Kernel recovery failed after {MAX_KERNEL_RECOVERY_ATTEMPTS + 1} attempts."
            ) from last_error

        return await self.code_executor.execute_code_blocks(code_blocks)

    @staticmethod
    def _is_recoverable_jupyter_error(error: RuntimeError) -> bool:
        message = str(error).lower()
        return (
            "kernel" in message
            or "did not become ready" in message
            or "wait_for_ready" in message
        )

    def workspace_root_dir(self):
        return self._base_session.workspace_root_dir()


@dataclass
class DebugTraceRecord:
    session_view: TempDebugSessionView
    chat_history: CellHistoryMemory
    last_debug_tag: Optional[str] = None
    last_debug_node: Optional[DebugNode] = None
    filter_tag: Optional[str] = None
    filter_node: Optional[PostDebuggingNode] = None


def create_debug_trace_record(session: Any) -> DebugTraceRecord:
    chat_history = clone_chat_history(session.chat_history)
    session_view = TempDebugSessionView(session, chat_history)
    return DebugTraceRecord(session_view=session_view, chat_history=chat_history)


def clone_debug_trace_record(
    trace_record: DebugTraceRecord,
    base_session: Optional[Any] = None,
) -> DebugTraceRecord:
    chat_history = clone_chat_history(trace_record.chat_history)
    rebound_session = base_session or trace_record.session_view._base_session
    session_view = TempDebugSessionView(rebound_session, chat_history)
    return DebugTraceRecord(
        session_view=session_view,
        chat_history=chat_history,
        last_debug_tag=trace_record.last_debug_tag,
        last_debug_node=trace_record.last_debug_node,
        filter_tag=trace_record.filter_tag,
        filter_node=trace_record.filter_node,
    )
