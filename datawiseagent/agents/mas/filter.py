from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from datawiseagent.memory.session import Session
from datawiseagent.memory.chat_history import CellHistoryMemory
from datawiseagent.common.types import (
    LLMResult,
    FormatType,
)
from datawiseagent.common.types.node import (
    PostDebuggingNode,
    CompletionUsage,
)
from datawiseagent.common.log import logger
from datawiseagent.prompts.datawise import (
    DATAWISE_AGENT_SYSTEM_PROMPT,
    MULTI_STAGE_AGENT_WORKFLOW_PROMPT,
    FILTER_ROLE_PROMPT,
    POST_DEBUGGING_APPEND_PROMPT,
    DEBUG_SUCCEED_TAG,
    DEBUG_FAIL_TAG,
)
from datawiseagent.coding.jupyter import JupyterCodeExecutor
from string import Template
import datawiseagent.agents.utils.execution as execution
import datawiseagent.agents.utils.parser as parser
import datawiseagent.agents.utils.prompt_tool as prompt_tool
from datawiseagent.agents.os.physical_agent import PhysicalAgent

if TYPE_CHECKING:
    from datawiseagent.agents.datawise_mas import DatawiseMAS
    from datawiseagent.agents.mas.debug_trace import DebugTraceRecord


def _should_rerun_before_filter(debug_trace: Optional["DebugTraceRecord"]) -> bool:
    if debug_trace is None:
        return True

    session_view = getattr(debug_trace, "session_view", None)
    base_session = getattr(session_view, "_base_session", None)

    # Concurrent isolated filter candidates are already rebuilt from the debug trace
    # before entering this stage. Replaying again would restart and rerun the same
    # branch history twice.
    return not getattr(base_session, "is_candidate_runtime", False)


async def self_debug_filter(
    physical_agent: PhysicalAgent,
    agent: "DatawiseMAS",
    session: Session,
    has_plan: bool = False,
    debug_trace: Optional["DebugTraceRecord"] = None,
) -> str:
    working_session = debug_trace.session_view if debug_trace is not None else session
    code_executor = working_session.code_executor
    assert (
        code_executor != None
    ), "Warning: The session's code_executor is None, cannot append code."
    assert isinstance(code_executor, JupyterCodeExecutor)
    chat_history = working_session.chat_history
    assert isinstance(chat_history, CellHistoryMemory)

    history = chat_history.to_messages(
        format=FormatType.PRESENT_CELLS, task_type="Post-debugging Stage"
    )
    if has_plan:
        system_prompt = (
            DATAWISE_AGENT_SYSTEM_PROMPT + MULTI_STAGE_AGENT_WORKFLOW_PROMPT
            + "\n"
            + FILTER_ROLE_PROMPT
        )
    else:
        system_prompt = DATAWISE_AGENT_SYSTEM_PROMPT + "\n" + FILTER_ROLE_PROMPT

    response: LLMResult = await physical_agent.llm.agenerate_response(
        prepend_prompt=[
            system_prompt,
            prompt_tool.get_workspace_prompt(working_session),
        ],
        history=history,
        append_prompt=POST_DEBUGGING_APPEND_PROMPT,
    )
    # logger.log_llm_result(response)

    generated_cells, tag = parser.parse_cell_response(
        response, end_tags=[DEBUG_SUCCEED_TAG, DEBUG_FAIL_TAG]
    )

    output_cells = await execution.execute_cells(
        working_session,
        generated_cells,
        rerun_all=_should_rerun_before_filter(debug_trace),
    )

    if execution.post_code_verification(output_cells):
        logger.info(
            "Code verification to cells filtered by the debugging phase passed successfully!"
        )
    else:
        logger.warn(
            "Code verification to cells filtered by the debugging phase failed to pass!"
        )

    node = PostDebuggingNode(
        cells_generated=generated_cells + output_cells,
        action_signal=tag,
        completion_usage=CompletionUsage(
            completion_tokens=response.recv_tokens,
            prompt_tokens=response.send_tokens,
            total_tokens=response.total_tokens,
        ),
    )
    if debug_trace is None:
        await agent._update_messages(
            session,
            node,
            action_signal=tag,
        )
    else:
        chat_history.add_messages(node, action_signal=tag)
        debug_trace.filter_tag = tag
        debug_trace.filter_node = node

    return tag, node
