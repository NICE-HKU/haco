from __future__ import annotations
from typing import List
from datawiseagent.memory.session import Session
from datawiseagent.common.types.cell import NotebookCell, CodeCell, CodeOutputCell

async def execute_cells(
    session: Session,
    generated_cells: List[NotebookCell],
    rerun_all: bool = False,
) -> List[NotebookCell]:

    if rerun_all:
        await session.rerun_cells()

    output_cells = []
    for cell in generated_cells:
        if isinstance(cell, CodeCell):
            cell_code_output = await session.safe_execute_code_blocks(
                cell.to_code_block()
            )
            cell.code_output = CodeOutputCell(code_result=cell_code_output)
            output_cells.append(cell.code_output)
    return output_cells

def post_code_verification(cells: List[NotebookCell]) -> bool:
    verification_result: bool = all(
        cell.code_result.exit_code == 0
        for cell in cells
        if isinstance(cell, CodeOutputCell)
    )
    return verification_result
