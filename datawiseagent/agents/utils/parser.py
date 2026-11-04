from __future__ import annotations
from typing import List, Tuple, Union

from datawiseagent.common.types import LLMResult, ConvertType
from datawiseagent.common.types.cell import NotebookCell, UserCell, StepCell, CodeCell
from datawiseagent.coding.code_utils import content_str

def parse_cell_response(
    response: LLMResult,
    end_tags: list[str],
) -> tuple[Union[NotebookCell, List[NotebookCell]], str]:
    generated_cells = NotebookCell.llm_result_convert(
        response, parse_mode=ConvertType.CONVERT_CELLS
    )

    content = content_str(response.content)
    if len(end_tags) == 0:
        raise ValueError("The parameter `end_tags` should not be empty!")

    remove_chars = "[]<>`"
    translator = str.maketrans("", "", remove_chars)

    strict2loose_end_tags = {}
    for tag in end_tags:
        cleaned_tag = tag.translate(translator)
        strict2loose_end_tags[tag] = cleaned_tag

    final_tag = end_tags[0]

    for tag, loose_tag in strict2loose_end_tags.items():
        if loose_tag in content:
            final_tag = tag
            break

    ablation = False
    new_generated_cells = []
    if ablation:
        for cell in generated_cells:
            if isinstance(cell, (UserCell, StepCell)):
                new_generated_cells.append(cell)
            elif isinstance(cell, CodeCell):
                code_content = str(cell.content)
                if not all(
                    line.strip() == "" or line.strip().startswith("#")
                    for line in code_content.splitlines()
                ):
                    new_generated_cells.append(cell)

        return new_generated_cells, final_tag

    return generated_cells, final_tag
