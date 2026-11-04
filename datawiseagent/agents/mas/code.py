from __future__ import annotations
from typing import TYPE_CHECKING
from string import Template

from datawiseagent.memory.session import Session
from datawiseagent.memory.chat_history import CellHistoryMemory
from datawiseagent.common.types import (
    LLMResult,
    FormatType,
)
from datawiseagent.common.types.node import (
    ExecutionNode,
    CompletionUsage,
)
from datawiseagent.common.log import logger
from datawiseagent.prompts.datawise import (
    DATAWISE_AGENT_SYSTEM_PROMPT,
    MULTI_STAGE_AGENT_WORKFLOW_PROMPT,
    CODER_ROLE_PROMPT,
    CURRENT_STEP_GOAL_PROMPT,
    APPEND_CODE_WITH_PLANNING_PROMPT,
    APPEND_CODE_WITHOUT_PLANNING_PROMPT,
    AWAIT_TAG,
    END_STEP_TAG,
    FULFILL_INSTRUCTION,
)
from datawiseagent.coding.code_utils import content_str
from datawiseagent.coding.jupyter import JupyterCodeExecutor
import datawiseagent.agents.utils.execution as execution
import datawiseagent.agents.utils.parser as parser
import datawiseagent.agents.utils.prompt_tool as prompt_tool
from datawiseagent.agents.os.physical_agent import PhysicalAgent

if TYPE_CHECKING:
    from datawiseagent.agents.datawise_mas import DatawiseMAS

# self_debug: bool = False,
# self_debug_max_number: int = 8,
def _response_contains_any_tag(response: LLMResult, end_tags: list[str]) -> bool:
    content = content_str(response.content)
    translator = str.maketrans("", "", "[]<>`")
    return any(tag.translate(translator) in content for tag in end_tags)


def _generated_cell_count(generated_cells) -> int:
    if isinstance(generated_cells, list):
        return len(generated_cells)
    return 0 if generated_cells is None else 1


def _log_coder_response(
    physical_agent: PhysicalAgent,
    response: LLMResult,
    parsed_tag: str,
    end_tags: list[str],
    generated_cells,
) -> None:
    logger.log_llm_result(response)
    logger.debug(
        (
            f"agent={physical_agent.name} parsed_action_signal={parsed_tag} "
            f"explicit_tag_found={_response_contains_any_tag(response, end_tags)} "
            f"generated_cell_count={_generated_cell_count(generated_cells)}"
        ),
        title="[Coder Parse]",
    )


async def append_code(
    physical_agent: PhysicalAgent,
    agent: "DatawiseMAS",
    session: Session,
    has_plan: bool = False,
) -> tuple[str, ExecutionNode]:
# ) -> tuple[str, str | None]:
    code_executor = session.code_executor
    assert (
        code_executor != None
    ), "Warning: The session's code_executor is None, cannot append code."

    assert isinstance(code_executor, JupyterCodeExecutor)
    chat_history = session.chat_history
    assert isinstance(chat_history, CellHistoryMemory)

    if has_plan:

        last_step_cell = chat_history.find_last_step()
        if last_step_cell:
            append_prompt = (
                Template(CURRENT_STEP_GOAL_PROMPT).safe_substitute(
                    {
                        "step_goal": last_step_cell.content,
                    }
                )
                + APPEND_CODE_WITH_PLANNING_PROMPT
            )
        else:
            append_prompt = APPEND_CODE_WITH_PLANNING_PROMPT

        response: LLMResult = await physical_agent.llm.agenerate_response(
            prepend_prompt=[
                DATAWISE_AGENT_SYSTEM_PROMPT
                + MULTI_STAGE_AGENT_WORKFLOW_PROMPT
                + "\n"
                + CODER_ROLE_PROMPT,
                prompt_tool.get_workspace_prompt(session),
            ],
            history=chat_history.to_messages(
                format=FormatType.PRESENT_CELLS,
                task_type="Incremental Execution Stage",
            ),
            append_prompt=append_prompt,
        )

        # parse the response
        generated_cells, append_code_tag = parser.parse_cell_response(
            response, end_tags=[AWAIT_TAG, END_STEP_TAG]
        )
        _log_coder_response(
            physical_agent,
            response,
            append_code_tag,
            [AWAIT_TAG, END_STEP_TAG],
            generated_cells,
        )
    else:
        response: LLMResult = await physical_agent.llm.agenerate_response(
            prepend_prompt=[
                DATAWISE_AGENT_SYSTEM_PROMPT + "\n" + CODER_ROLE_PROMPT,
                prompt_tool.get_workspace_prompt(session),
            ],
            history=chat_history.to_messages(
                format=FormatType.PRESENT_CELLS,
                task_type="Incremental Execution Stage",
            ),
            append_prompt=APPEND_CODE_WITHOUT_PLANNING_PROMPT,
        )

        # parse the response
        generated_cells, append_code_tag = parser.parse_cell_response(
            response,
            end_tags=[AWAIT_TAG, FULFILL_INSTRUCTION],
        )
        _log_coder_response(
            physical_agent,
            response,
            append_code_tag,
            [AWAIT_TAG, FULFILL_INSTRUCTION],
            generated_cells,
        )

    # execute all the code cells
    output_cells = await execution.execute_cells(session, generated_cells)
    execution_node = ExecutionNode(
        cells_generated=generated_cells + output_cells,
        action_signal=append_code_tag,
        completion_usage=CompletionUsage(
            completion_tokens=response.recv_tokens,
            prompt_tokens=response.send_tokens,
            total_tokens=response.total_tokens,
        ),
    )
    # chat_history.add_messages(execution_node, action_signal=append_code_tag)
    # Committed by the outer FSM/orchestrator to avoid double-adding the same node.
    # await agent._update_messages(
    #     session, execution_node, action_signal=append_code_tag
    # )

    # debug_tag = None
    # if self_debug and not execution.post_code_verification(output_cells):
    #     # Self-debug enabled.
    #     # Any of the code cells doesn't pass the code verification, or raises error during execution.
    #     debug_tag = await agent._self_debug(
    #         session, execution_node, self_debug_max_number, has_plan=has_plan
    #     )
    # return append_code_tag, debug_tag
    return append_code_tag, execution_node
