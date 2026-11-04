from __future__ import annotations
from typing import TYPE_CHECKING
from string import Template

from datawiseagent.memory.session import Session
from datawiseagent.memory.chat_history import CellHistoryMemory
from datawiseagent.common.types import (
    LLMResult,
    FormatType,
    ConvertType,
)
from datawiseagent.common.types.cell import (
    NotebookCell,
)
from datawiseagent.common.types.node import (
    StepNode,
    CompletionUsage,
)
from datawiseagent.common.log import logger
from datawiseagent.prompts.datawise import (
    DATAWISE_AGENT_SYSTEM_PROMPT,
    MULTI_STAGE_AGENT_WORKFLOW_PROMPT,
    PLANNER_ROLE_PROMPT,
    AWAIT_TAG,
    INITIATE_WITH_PLANNING_PROMPT,
    ITERATE_ON_LAST_STEP,
    ADVANCE_TO_NEXT_STEP,
    FULFILL_INSTRUCTION,
    PLANNING_APPEND_PROMPT,
    CURRENT_STEP_GOAL_PROMPT,
    CURRENT_USER_INSTRCUTION_PROMPT,
)
from datawiseagent.coding.jupyter import JupyterCodeExecutor
import datawiseagent.agents.utils.execution as execution
import datawiseagent.agents.utils.parser as parser
import datawiseagent.agents.utils.prompt_tool as prompt_tool
from datawiseagent.agents.os.physical_agent import PhysicalAgent


if TYPE_CHECKING:
    from datawiseagent.agents.datawise_mas import DatawiseMAS


def _log_planner_llm_result(
    physical_agent: PhysicalAgent,
    response: LLMResult,
    planner_task_id: str | None = None,
    planner_step_id: int | None = None,
    planner_small_step_id: int | None = None,
) -> None:
    parts = [f"planner={physical_agent.name}"]
    if planner_task_id is not None:
        parts.append(f"task_id={planner_task_id}")
    if planner_step_id is not None:
        parts.append(f"step_id={planner_step_id}")
    if planner_small_step_id is not None:
        parts.append(f"small_step_id={planner_small_step_id}")
    logger.info(f"[PLANNER_OUTPUT] {' | '.join(parts)}")
    logger.log_llm_result(response)



# self_debug: bool = True, 
# self_debug_max_number: int = 8
async def initiate_step(
    physical_agent: PhysicalAgent,
    agent: "DatawiseMAS", 
    session: Session,
    planner_task_id: str | None = None,
    planner_step_id: int | None = None,
    planner_small_step_id: int | None = None,
):
    code_executor = session.code_executor
    assert (
        code_executor != None
    ), "Warning: The session's code_executor is None, cannot append code."
    assert isinstance(code_executor, JupyterCodeExecutor)
    chat_history = session.chat_history
    assert isinstance(chat_history, CellHistoryMemory)

    response: LLMResult = await physical_agent.llm.agenerate_response(
        prepend_prompt=[
            DATAWISE_AGENT_SYSTEM_PROMPT
            + MULTI_STAGE_AGENT_WORKFLOW_PROMPT
            + "\n"
            + PLANNER_ROLE_PROMPT,
            prompt_tool.get_workspace_prompt(session),
        ],
        history=chat_history.to_messages(format=FormatType.PRESENT_CELLS),
        append_prompt=INITIATE_WITH_PLANNING_PROMPT,
    )
    _log_planner_llm_result(
        physical_agent,
        response,
        planner_task_id=planner_task_id,
        planner_step_id=planner_step_id,
        planner_small_step_id=planner_small_step_id,
    )

    # parse the response
    generated_cells = NotebookCell.llm_result_convert(
        response, parse_mode=ConvertType.CONVERT_CELLS
    )

    # execute all the code cells
    output_cells = await execution.execute_cells(session, generated_cells)

    step_node = StepNode(
        cells_generated=generated_cells + output_cells,
        action_signal=AWAIT_TAG,
        completion_usage=CompletionUsage(
            completion_tokens=response.recv_tokens,
            prompt_tokens=response.send_tokens,
            total_tokens=response.total_tokens,
        ),
    )
    # chat_history.add_messages(step_node, action_signal=AWAIT_TAG)
    # Committed by the outer FSM/orchestrator to avoid double-adding the same node.
    # await agent._update_messages(session, step_node, action_signal=AWAIT_TAG)

    # debug_tag = None
    # if self_debug and not execution.post_code_verification(output_cells):
    #     # Self-debug enabled.
    #     # Any of the code cells doesn't pass the code verification, or raises error during execution.

    #     debug_tag = await agent._self_debug(
    #         session, step_node, self_debug_max_number, has_plan=True
    #     )

    # Ignore the `debug_tag` at first
    # return debug_tag
    return AWAIT_TAG, step_node

# self_debug: bool = True, 
# self_debug_max_number: int = 8
async def update_plan(
    physical_agent: PhysicalAgent,
    agent: "DatawiseMAS", 
    session: Session,
    planner_task_id: str | None = None,
    planner_step_id: int | None = None,
    planner_small_step_id: int | None = None,
):
    code_executor = session.code_executor
    assert (
        code_executor != None
    ), "Warning: The session's code_executor is None, cannot append code."
    assert isinstance(code_executor, JupyterCodeExecutor)
    chat_history = session.chat_history
    assert isinstance(chat_history, CellHistoryMemory)

    last_step_cell = chat_history.find_last_step()
    if last_step_cell:
        append_prompt = (
            Template(CURRENT_USER_INSTRCUTION_PROMPT).safe_substitute(
                {
                    "user_query": session.current_user_query,
                }
            )
            + Template(CURRENT_STEP_GOAL_PROMPT).safe_substitute(
                {
                    "step_goal": last_step_cell.content,
                }
            )
            + PLANNING_APPEND_PROMPT
        )
    else:
        append_prompt = PLANNING_APPEND_PROMPT

    response: LLMResult = await physical_agent.llm.agenerate_response(
        prepend_prompt=[
            DATAWISE_AGENT_SYSTEM_PROMPT
            + MULTI_STAGE_AGENT_WORKFLOW_PROMPT
            + "\n"
            + PLANNER_ROLE_PROMPT,
            prompt_tool.get_workspace_prompt(session),
        ],
        history=chat_history.to_messages(
            format=FormatType.PRESENT_CELLS, task_type="Planning Stage"
        ),
        append_prompt=append_prompt,
    )
    _log_planner_llm_result(
        physical_agent,
        response,
        planner_task_id=planner_task_id,
        planner_step_id=planner_step_id,
        planner_small_step_id=planner_small_step_id,
    )

    generated_cells, planning_tag = parser.parse_cell_response(
        response,
        end_tags=[ITERATE_ON_LAST_STEP, ADVANCE_TO_NEXT_STEP, FULFILL_INSTRUCTION],
    )

    step_node = None # debug

    if planning_tag == ITERATE_ON_LAST_STEP:
        # add dummy node to the right of the last step
        chat_history.clear_last_step()
        output_cells = await execution.execute_cells(
            session, generated_cells, rerun_all=True
        )

        step_node = StepNode(
            cells_generated=generated_cells + output_cells,
            action_signal=ITERATE_ON_LAST_STEP,
            completion_usage=CompletionUsage(
                completion_tokens=response.recv_tokens,
                prompt_tokens=response.send_tokens,
                total_tokens=response.total_tokens,
            ),
        )

        # chat_history.add_messages(step_node, action_signal=ITERATE_ON_LAST_STEP)
        # Committed by the outer FSM/orchestrator to avoid double-adding the same node.
        # await agent._update_messages(
        #     session, step_node, action_signal=ITERATE_ON_LAST_STEP
        # )

        # debug_tag = None
        # if self_debug and not execution.post_code_verification(output_cells):
        #     # Self-debug enabled.
        #     # Any of the code cells doesn't pass the code verification, or raises error during execution.
        #     debug_tag = await agent._self_debug(
        #         session, step_node, self_debug_max_number, has_plan=True
        #     )

    elif planning_tag == ADVANCE_TO_NEXT_STEP:
        output_cells = await execution.execute_cells(session, generated_cells)
        step_node = StepNode(
            cells_generated=generated_cells + output_cells,
            action_signal=ADVANCE_TO_NEXT_STEP,
            completion_usage=CompletionUsage(
                completion_tokens=response.recv_tokens,
                prompt_tokens=response.send_tokens,
                total_tokens=response.total_tokens,
            ),
        )
        # chat_history.add_messages(step_node, action_signal=ADVANCE_TO_NEXT_STEP)
        # Committed by the outer FSM/orchestrator to avoid double-adding the same node.
        # await agent._update_messages(
        #     session, step_node, action_signal=ADVANCE_TO_NEXT_STEP
        # )

        # debug_tag = None
        # if self_debug and not execution.post_code_verification(output_cells):
        #     # Self-debug enabled.
        #     # Any of the code cells doesn't pass the code verification, or raises error during execution.
        #     debug_tag = await agent._self_debug(
        #         session, step_node, self_debug_max_number, has_plan=True
        #     )

    elif planning_tag == FULFILL_INSTRUCTION:
        step_node = StepNode(
            cells_generated=generated_cells,
            action_signal=FULFILL_INSTRUCTION,
            completion_usage=CompletionUsage(
                completion_tokens=response.recv_tokens,
                prompt_tokens=response.send_tokens,
                total_tokens=response.total_tokens,
            ),
        )
        # Committed by the outer FSM/orchestrator to avoid double-adding the same node.
        # await agent._update_messages(
        #     session,
        #     step_node,
        #     action_signal=FULFILL_INSTRUCTION,
        # )

    # return planning_tag, debug_tag
    return planning_tag, step_node # ✅ 统一返回结果
