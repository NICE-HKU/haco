

from typing import Any, Dict, Optional


PLANNER_INITIAL_RUBRIC = (
    "- This is the initial planning step. Reward responses that initiate exactly one well-scoped first step aligned with the user's goal.\n"
    "- Reward first steps that are concrete, executable, and easy for the execution stage to implement and verify.\n"
    "- Reward lightweight inspection or bootstrap code only when it materially helps define, validate, or safely launch the first step.\n"
    "- Reward clear handoff quality: the planner should leave the coder with a precise, actionable next step rather than a vague intention.\n"
    "- Do not penalize the planner for not solving the entire task or for not providing a full multi-step decomposition in this initial step.\n"
    "- Do not reward over-implementation that bypasses the execution stage by doing the substantive solution when the planner should only launch the first step.\n"
    "- Penalize first steps that are too broad, too vague, operationally risky, or disconnected from the user's actual objective.\n"
    "- Penalize avoidable assumptions that make the first step brittle when a lightweight inspection would have been safer."
)


PLANNER_UPDATE_RUBRIC = (
    "- This is a follow-up planning step after prior execution or debugging work.\n"
    "- Reward correct evaluation of the current step outcome and the right planning decision: iterate on the current step, advance to the next step, or fulfill the instruction.\n"
    "- Reward next steps that are actionable, well-prioritized, and clearly aligned with the user's goal and the current notebook state.\n"
    "- Reward lightweight inspection, validation, or bootstrap code only when it directly supports the planning decision or prepares a clean handoff to execution.\n"
    "- If the output is code: reward syntactically correct, logically sound code that directly supports the current planning decision; penalize pseudo-code, syntax errors, or code that does not support the stated sub-task.\n"
    "- If the output is mixed (plan + code): evaluate both parts — the planning decision for clarity/correctness and the code for usefulness/correctness.\n"
    "- Do not penalize for brevity if the output is complete and correct for this planning step.\n"
    "- Penalize outputs that are internally inconsistent, operationally hard to execute, or fail to address the current planning decision.\n"
    "- Penalize planner outputs that drift into full implementation, redefine the task scope unnecessarily, or leave the coder with no clear next action."
)


PLANNER_INITIAL_STAGE_GUIDANCE = (
    "This is the **initial planner** step. Judge whether the agent selected a strong first step, provided a clean handoff to execution, and stayed within planner-bootstrapper responsibilities. "
    "Do NOT require the planner to complete the entire user task, and do NOT reward it for unnecessarily taking over the execution stage."
)


PLANNER_UPDATE_STAGE_GUIDANCE = (
    "This is a **follow-up planner** step. Judge whether the agent correctly interpreted prior progress, chose the right next planning action "
    "(iterate, advance, or fulfill), and used any code only to support that decision or bootstrap the next execution step."
)


                                  
ROLE_RUBRICS: Dict[str, str] = {
    "planner": (
        "- The planner is a planner-bootstrapper: it may produce a planning decision, lightweight executable code, or a mix of both — evaluate based on what was actually produced.\n"
        "- If the output is a plan: reward actionable, well-prioritized next steps clearly aligned with the user's goal and current notebook state; penalize vague, incomplete, or inconsistent planning.\n"
        "- If the output includes code: reward lightweight inspection, validation, or bootstrap code that helps planning or creates a clean handoff to the coder; penalize pseudo-code, syntax errors, or code that does not support the planning decision.\n"
        "- If the output is mixed (plan + code): evaluate both parts — the planning decision for clarity and the code for usefulness.\n"
        "- Do not penalize for brevity if the output is complete and correct for this step.\n"
        "- Penalize outputs that are internally inconsistent, operationally hard to execute, or fail to address the step's goal.\n"
        "- Penalize over-implementation that effectively completes the coder's main job instead of selecting and safely launching the next step."
    ),
    "coder": (
        "- Evaluate the coder on three dimensions: Execution Correctness, Marginal Progress, and Transition Quality.\n"
        "- Reward code that is syntactically correct, logically sound, runnable as-is, and directly solves the stated sub-task when new execution is actually needed.\n"
        "- Reward responses that add genuine new progress beyond the previous committed result, such as new computation, new validation, or a necessary refinement.\n"
        "- Reward a reasonable end-of-step decision when the current step is already satisfied by existing verified notebook state or by the response itself.\n"
        "- Penalize pseudo-code, syntax errors, runtime risks, missing key logic, or code that does not address the sub-task.\n"
        "- Penalize redundant re-execution that mostly repeats the previous committed result without adding meaningful new progress, even if the repeated code is correct.\n"
        "- Penalize choosing <await> when the current step is already complete and the response adds no meaningful new value.\n"
        "- If the current step still requires new execution and the response contains no executable code at all, the score should not exceed 5.0.\n"
        "- CRITICAL: If the code runs but produces incorrect output or fails to complete the stated task (e.g., wrong column name, wrong result), treat this as a major failure — score should be 4.0 or below.\n"
        "- CRITICAL: If the task explicitly requires a specific algorithm/model/method and the code uses a different one (e.g., LogisticRegression instead of LinearRegression), this is a major requirement violation — deduct at least 3 points from what the code would otherwise score.\n"
        "- Maintain consistent scoring standards across steps: the same type of error should receive the same penalty regardless of which step it appears in."
    ),
    "debugger": (
        "- Reward accurate root-cause diagnosis that correctly identifies the source of the error or failure.\n"
        "- Reward minimal, targeted fixes that resolve the issue without introducing new risks.\n"
        "- Reward verification that the proposed repair actually addresses the failure.\n"
        "- Reward clear explanation of why the fix works.\n"
        "- Penalize symptom-only explanations that do not address the root cause.\n"
        "- Penalize broad rewrites, unnecessary scope expansion, or risky edits that change unrelated code or may introduce regressions."
    ),
    "filter": (
        "- This role is a post-debug cleaner, not a general reviewer.\n"
        "- If debugging succeeded, reward outputs that extract clean, self-contained, complete replacement code that removes noisy intermediate attempts.\n"
        "- If debugging failed, reward concise, credible diagnostic reports that explain what was tried, what was learned, and why the issue remains unresolved.\n"
        "- Reward outputs that preserve the useful end result of debugging while discarding exploratory clutter.\n"
        "- Penalize outputs that continue open-ended debugging, leave partial intermediate attempts in place, or produce a cleaned artifact that is incomplete or not self-contained."
    ),
    "default": (
        "- Reward correctness, completeness, and relevance to the query.\n"
        "- Penalize factual errors, omissions, and off-topic responses."
    ),
}


SCORE_ANCHORS = (
    "- 10: Fully correct, complete, and clearly justified.\n"
    "- 8: Strong overall quality, only minor issues.\n"
    "- 6: Partially correct or missing important details.\n"
    "- 4: Major problems, low reliability.\n"
    "- 2: Mostly incorrect or unusable."
)


                                
_ROLE_PROMPT_TEMPLATES: Dict[str, str] = {
    "planner": """\
You are an expert evaluator assessing a **planner agent** response.

**Overall Task Goal:**
{query}

**Evaluation Context:**
{step_context}

{judge_context_block}\
**Agent Response:**
{responses}

**Evaluation Criteria (Planner-specific):**
{rubric}

**Planner Stage Guidance:**
{planner_stage_guidance}

**Score Anchors:**
{score_anchors}

**Important:** The planner is acting as a planner-bootstrapper. It may produce a planning decision, lightweight executable code, or both. First identify whether it selected the right next step, then judge whether any code genuinely supports planning or bootstrap handoff. Do NOT reward the planner for taking over the coder's substantive implementation work.

**Output Rules:**
- Return ONLY valid JSON (no markdown, no prose).
- score must be a number in [0.0, 10.0] with exactly one decimal place (e.g., 7.4).
- Unless two responses are substantively equivalent, use at least 0.1 difference.
- Keep reasoning concise (<= 40 words).

[
  {{
    "agent_name": "name-of-the-agent",
    "score": 8.0,
    "reasoning": "short reason"
  }}
]
""",
    "coder": """\
You are an expert evaluator assessing a **coder agent** response.

**Overall Task Goal:**
{query}

**Evaluation Context:**
{step_context}

{judge_context_block}\
**Agent Response (code produced for this step):**
{responses}

**Evaluation Criteria (Coder-specific):**
{rubric}

**Score Anchors:**
{score_anchors}

**Important:** Evaluate ONLY this specific step. Use the additional step state, action signal, and previous committed result when provided. Judge not only whether the code is correct, but also whether this response adds meaningful progress and whether the continue/end decision is appropriate.

**Hard Rules (apply before scoring):**
1. If the current step still requires new execution and the response has no executable code, score ≤ 5.0.
2. If the response mostly repeats previously completed work with little or no new progress, cap the score around 6.0 or below depending on severity.
3. If the step is already complete but the agent still chooses <await> without meaningful new value, deduct for poor transition quality.
4. If the agent reasonably chooses to end the step because the goal is already satisfied by existing verified state, do NOT penalize solely for not repeating code.
5. Code runs but produces wrong output or fails the task → score ≤ 4.0.
6. Task requires a specific algorithm/model/method but a different one is used → deduct ≥ 3 points from the base score.
7. Apply the same penalty for the same type of error consistently across all steps.

**Output Rules:**
- Return ONLY valid JSON (no markdown, no prose).
- score must be a number in [0.0, 10.0] with exactly one decimal place (e.g., 7.4).
- Unless two responses are substantively equivalent, use at least 0.1 difference.
- Keep reasoning concise (<= 40 words). Do NOT repeat words or phrases.

[
  {{
    "agent_name": "name-of-the-agent",
    "score": 8.0,
    "reasoning": "short reason"
  }}
]
""",
    "debugger": """\
You are an expert evaluator assessing a **debugger agent** response.

**Overall Task Goal:**
{query}

**Evaluation Context:**
{step_context}

{judge_context_block}\
**Agent Response (debug/fix produced for this step):**
{responses}

**Evaluation Criteria (Debugger-specific):**
{rubric}

**Score Anchors:**
{score_anchors}

**Important:** Evaluate ONLY the diagnosis and repair work produced for this specific step. Judge whether the root cause was correctly identified, whether the fix is minimal and targeted, and whether the response provides enough evidence that the repair was actually validated.

**Output Rules:**
- Return ONLY valid JSON (no markdown, no prose).
- score must be a number in [0.0, 10.0] with exactly one decimal place (e.g., 7.4).
- Unless two responses are substantively equivalent, use at least 0.1 difference.
- Keep reasoning concise (<= 40 words).

[
  {{
    "agent_name": "name-of-the-agent",
    "score": 8.0,
    "reasoning": "short reason"
  }}
]
""",
    "filter": """\
You are an expert evaluator assessing a **filter agent** response.

**Overall Task Goal:**
{query}

**Evaluation Context:**
{step_context}

{judge_context_block}\
**Agent Response (post-debug cleaning result for this step):**
{responses}

**Evaluation Criteria (Filter-specific):**
{rubric}

**Score Anchors:**
{score_anchors}

**Important:** Evaluate ONLY the post-debug cleaning work done in this specific step. If debugging succeeded, judge whether the response extracts a clean replacement artifact. If debugging failed, judge whether it produces a concise, credible diagnostic report. Do NOT score this role as a general reviewer, validator, or safety checker.

**Output Rules:**
- Return ONLY valid JSON (no markdown, no prose).
- score must be a number in [0.0, 10.0] with exactly one decimal place (e.g., 7.4).
- Unless two responses are substantively equivalent, use at least 0.1 difference.
- Keep reasoning concise (<= 40 words).

[
  {{
    "agent_name": "name-of-the-agent",
    "score": 8.0,
    "reasoning": "short reason"
  }}
]
""",
    "default": """\
You are an expert evaluator assessing an agent response.

**Overall Task Goal:**
{query}

**Evaluation Context:**
{step_context}

{judge_context_block}\
**Agent Response:**
{responses}

**Evaluation Criteria:**
{rubric}

**Score Anchors:**
{score_anchors}

**Output Rules:**
- Return ONLY valid JSON (no markdown, no prose).
- score must be a number in [0.0, 10.0] with exactly one decimal place (e.g., 7.4).
- Unless two responses are substantively equivalent, use at least 0.1 difference.
- Keep reasoning concise (<= 40 words).

[
  {{
    "agent_name": "name-of-the-agent",
    "score": 8.0,
    "reasoning": "short reason"
  }}
]
""",
}


def _truncate_for_prompt(value: Any, max_chars: int = 1600) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "null"}:
        return ""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated)"


def _parse_step_info(step_info: str) -> Dict[str, str]:
    if not step_info:
        return {}

    parsed: Dict[str, str] = {}
    for part in step_info.split("|"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _is_initial_planner_step(
    role: str,
    step_info: str,
    judge_context: Optional[Dict[str, Any]],
) -> bool:
    if role != "planner":
        return False

    step_fields = _parse_step_info(step_info)
    if step_fields.get("step_id") != "1":
        return False

    current_action_signal = (judge_context or {}).get("current_action_signal")
    if current_action_signal and current_action_signal != "<await>":
        return False

    return True


def _format_judge_context(judge_context: Optional[Dict[str, Any]]) -> str:
    if not judge_context:
        return ""

    sections = []
    field_order = [
        ("current_step_goal", "Current Step Goal"),
        ("current_action_signal", "Current Action Signal"),
        ("previous_winner_action_signal", "Previous Winner Action Signal"),
        ("previous_winner_response", "Previous Winner Response"),
        ("history_summary", "Existing Notebook State Summary"),
        ("redundancy_hint", "Redundancy Hint"),
    ]
    for key, label in field_order:
        value = judge_context.get(key)
        formatted_value = _truncate_for_prompt(value)
        if formatted_value:
            sections.append(f"**{label}:**\n{formatted_value}")

    if not sections:
        return ""

    return "**Additional Step State:**\n" + "\n\n".join(sections) + "\n\n"


def build_scoring_prompt(
    role: str,
    query: str,
    responses: str,
    step_info: str = "",
    judge_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a role-aware scoring prompt for the LLM judge.

    Args:
        role: Agent role (planner / coder / debugger / filter / default).
        query: The overall task goal or user query.
        responses: Formatted agent response(s) to evaluate.
        step_info: Optional context describing which step this is (e.g. "Task 2, Step 3/5").

    Returns:
        A fully formatted prompt string ready to send to the LLM judge.
    """
    rubric = ROLE_RUBRICS.get(role, ROLE_RUBRICS["default"])
    template = _ROLE_PROMPT_TEMPLATES.get(role, _ROLE_PROMPT_TEMPLATES["default"])
    planner_stage_guidance = ""

    if role == "planner":
        if _is_initial_planner_step(role, step_info, judge_context):
            rubric = PLANNER_INITIAL_RUBRIC
            planner_stage_guidance = PLANNER_INITIAL_STAGE_GUIDANCE
        else:
            rubric = PLANNER_UPDATE_RUBRIC
            planner_stage_guidance = PLANNER_UPDATE_STAGE_GUIDANCE

    if step_info:
        step_context = (
            f"{step_info}\n"
            "This is one step in a potentially multi-step task (it may be the first, an intermediate, or the final step). "
            "Evaluate ONLY how well this agent performed THIS specific step — do NOT penalize for not completing the entire task."
        )
    else:
        step_context = "Evaluate the quality of this agent's response for the current step."

    return template.format(
        query=query,
        step_context=step_context,
        judge_context_block=_format_judge_context(judge_context),
        responses=responses,
        rubric=rubric,
        planner_stage_guidance=planner_stage_guidance,
        score_anchors=SCORE_ANCHORS,
    )
