from __future__ import annotations
from string import Template
from datawiseagent.memory.session import Session
from datawiseagent.prompts.datawise import CURRENT_WORKSPACE_STRUCTURE_AND_STATUS

def get_workspace_prompt(session: Session) -> str:
    """
    Generates the workspace prompt string with the current status.
    """
    return Template(CURRENT_WORKSPACE_STRUCTURE_AND_STATUS).safe_substitute(
        {
            "workspace_root_dir": session.workspace_root_dir(),
            "workspace_status": session.workspace.fetch_workspace_status(),
        }
    )
