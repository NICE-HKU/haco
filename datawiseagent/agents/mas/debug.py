from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from datawiseagent.memory.session import Session
from datawiseagent.memory.chat_history import CellHistoryMemory
from datawiseagent.common.types import (
    LLMResult,
    FormatType,
)
from datawiseagent.common.types.node import (
    ExecutionNode,
    StepNode,
    DebugNode,
    CompletionUsage,
)
from datawiseagent.common.types.cell import (
    MarkdownCell,
    CodeCell,
    CodeOutputCell,
)
from datawiseagent.common.log import logger
from datawiseagent.prompts.datawise import (
    DATAWISE_AGENT_SYSTEM_PROMPT,
    MULTI_STAGE_AGENT_WORKFLOW_PROMPT,
    DEBUGGER_ROLE_PROMPT,
    CURRENT_STEP_GOAL_PROMPT,
    DEBUGGING_APPEND_PROMPT,
    AWAIT_TAG,
    END_DEBUG_TAG,
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

def filter_safe_cells(
    session: Session, current_node: Optional[ExecutionNode | StepNode]
):
    if current_node is None:
        logger.warn("filter_safe_cells received None current_node. Skipping filter.")
        return []

    # Filter the safe cells in the beginning to update the chat history
    init_generated_cells = current_node.cells_generated
    safe_generated_cells = []
    safe_output_cells = []
    generated_cells_to_debug = []
    output_cells_to_debug = []
    right_tag = True
    for cell in init_generated_cells:
        if isinstance(cell, MarkdownCell):
            if right_tag:
                safe_generated_cells.append(cell)
            else:
                generated_cells_to_debug.append(cell)
        elif isinstance(cell, CodeCell):
            if execution.post_code_verification([cell.code_output]):
                if right_tag:
                    safe_generated_cells.append(cell)
                    safe_output_cells.append(cell.code_output)
                else:
                    generated_cells_to_debug.append(cell)
                    output_cells_to_debug.append(cell.code_output)
            else:
                right_tag = False
                generated_cells_to_debug.append(cell)
                output_cells_to_debug.append(cell.code_output)

    current_node.correct_cells = safe_generated_cells + safe_output_cells
    current_node.cells_to_debug = generated_cells_to_debug + output_cells_to_debug

    # session.chat_history.add_messages()
    return generated_cells_to_debug + output_cells_to_debug

async def self_debug_execution(
    physical_agent: PhysicalAgent,
    agent: "DatawiseMAS",
    session: Session,
    max_turn_number: int = 8,
    has_plan: bool = False,
    debug_trace: Optional["DebugTraceRecord"] = None,
):
    working_session = debug_trace.session_view if debug_trace is not None else session
    code_executor = working_session.code_executor
    assert (
        code_executor != None
    ), "Warning: The session's code_executor is None, cannot append code."
    assert isinstance(code_executor, JupyterCodeExecutor)
    chat_history = working_session.chat_history
    assert isinstance(chat_history, CellHistoryMemory)

    tag = AWAIT_TAG
    turn_number = 0
    last_node = None
    while not (tag == END_DEBUG_TAG or turn_number >= max_turn_number):

        history = chat_history.to_messages(
            format=FormatType.PRESENT_CELLS, task_type="Debugging Stage"
        )

        last_step_cell = chat_history.find_last_step()

        if has_plan:
            system_prompt = (
                DATAWISE_AGENT_SYSTEM_PROMPT + MULTI_STAGE_AGENT_WORKFLOW_PROMPT
                + "\n"
                + DEBUGGER_ROLE_PROMPT
            )
        else:
            system_prompt = DATAWISE_AGENT_SYSTEM_PROMPT + "\n" + DEBUGGER_ROLE_PROMPT

        if last_step_cell:
            append_prompt = (
                Template(CURRENT_STEP_GOAL_PROMPT).safe_substitute(
                    {
                        "step_goal": last_step_cell.content,
                    }
                )
                + DEBUGGING_APPEND_PROMPT
            )
        else:
            append_prompt = DEBUGGING_APPEND_PROMPT

        response: LLMResult = await physical_agent.llm.agenerate_response(
            prepend_prompt=[
                system_prompt,
                prompt_tool.get_workspace_prompt(working_session),
            ],
            history=history,
            append_prompt=append_prompt,
        )
        # logger.log_llm_result(response)

        generated_cells, tag = parser.parse_cell_response(
            response, end_tags=[AWAIT_TAG, END_DEBUG_TAG]
        )

        output_cells = await execution.execute_cells(working_session, generated_cells)

        node = DebugNode(
            cells_generated=generated_cells + output_cells,
            action_signal=tag,
            completion_usage=CompletionUsage(
                completion_tokens=response.recv_tokens,
                prompt_tokens=response.send_tokens,
                total_tokens=response.total_tokens,
            ),
        )
        last_node = node
        if debug_trace is None:
            await agent._update_messages(
                session,
                node,
                action_signal=tag,
            )
        else:
            chat_history.add_messages(node, action_signal=tag)
            debug_trace.last_debug_tag = tag
            debug_trace.last_debug_node = node
        turn_number += 1

    return tag, last_node
