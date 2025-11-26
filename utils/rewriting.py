# rewriting.py

from openai import OpenAI, RateLimitError
from config import OPENAI_API_KEY
import re

client = OpenAI(api_key=OPENAI_API_KEY)


def _normalize_ws(s: str) -> str:
    """Collapse internal whitespace, keep single newlines"""
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\r\n?", "\n", s)
    return s.strip()


def _strip_code_fences(s: str) -> str:
    """Remove markdown code fences if present"""
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        s = "\n".join(line for line in lines if not line.strip().startswith("```"))
    return s.strip()


def _truncate_middle(text: str, max_chars: int = 24000) -> str:
    """Keep first and last parts; drop middle if too long"""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return text[:head].rstrip() + "\n\n[...middle truncated...]\n\n" + text[-tail:].lstrip()


def rewrite_abstract(full_text, original_abstract):
    """
    Uses GPT to rewrite an abstract into two structured versions (250-word and 300-word).
    Follows the updated CONSORT guidelines specification.
    Returns: (rewritten_250, rewritten_300)
    """

    # Prepare inputs safely
    full_text = _normalize_ws(full_text or "")
    original_abstract = _normalize_ws(original_abstract or "")

    if not full_text:
        full_text = "No full text available."
    truncated_full = _truncate_middle(full_text, max_chars=24000)

    # PROTOCOL-COMPLIANT PROMPT (Updated to match new requirements)
    prompt = f"""You are an experienced researcher writing a scientific manuscript for a surgical journal.

Below is the full manuscript including an abstract.

Your aim is to re-write the abstract so that it is compliant with CONSORT guidelines for randomised controlled trials. Use the attached manuscript to access information that is missing from the current abstract.

{{Full Article}}
{truncated_full}

{{Original Abstract}}
{original_abstract}

The abstract should be structured with the following sections: Background, Methods, Results, Interpretation.

Ensure your abstract includes all the following information:
- Title: Identify the study as a randomised controlled trial.
- Trial design: State the unit of randomisation and describe the sample size calculation.
- Participants: State both the participant eligibility criteria and the study setting (i.e. where the study was completed).
- Intervention: Describe both the intervention(s) and the comparator (control).
- Objective: Describe the study objective.
- Outcome: State and define the primary outcome, including the timeframe over which it was measured.
- Randomisation: State both randomisation method and allocation concealment method.
- Blinding (masking): State which of the outcome assessors, patient, and clinician were blinded.
- Number randomised: State the total number of participants randomised to each group and specify the period of recruitment (for example, 200 participants were randomised: 100 to the intervention and 100 to the control group, between January 2020 and June 2021).
- Number analysed: State the number analysed in each group and whether the analysis was intention-to-treat, per-protocol, or another approach (for example, 98 participants in the intervention group and 97 in the control group were analysed using an intention-to-treat approach).
- Results: Report the main outcome's effect size and its precision (for example, mean difference = 5.2 mm Hg, 95% CI 3.1 to 7.3; p = 0.01).
- Harm: Describe whether adverse events occurred, their frequency and severity by group, and state if none were reported (for example, 3/100 [3%] in the intervention group and 1/98 [1%] in the control group experienced mild gastrointestinal side-effects).
- Trial registration: Provide the trial registration number on the Trial Register.
- Funding: Report the source of funding for the trial.

Rewrite the abstract into two SEPARATE versions:
1. A 250-word version
2. A 300-word version

Separate the versions using this marker: ---

Always return integers (not strings).
Do NOT include explanations, comments, markdown, or formatting.
Only use information from the existing abstract and full-text manuscript. Do not make any details up.
"""

    def _call(model_name: str):
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1400,
        )
        return resp.choices[0].message.content or ""

    # Try primary model, fallback to gpt-4o on rate limit
    used_model = "gpt-5"
    try:
        content = _call(used_model)
    except RateLimitError:
        used_model = "gpt-4o"
        content = _call(used_model)
    except Exception:
        used_model = "gpt-4o"
        content = _call(used_model)

    content = _strip_code_fences(content)

    # Parse using the protocol's specified delimiter: ---
    if "---" in content:
        parts = content.split("---", maxsplit=1)
        v250 = _normalize_ws(parts[0])
        v300 = _normalize_ws(parts[1]) if len(parts) > 1 else _normalize_ws(parts[0])
    else:
        # Fallback: if no delimiter, duplicate the output
        print("⚠️ No '---' delimiter found. Using same text for both versions.")
        v250 = _normalize_ws(content)
        v300 = _normalize_ws(content)

    # If parsing completely failed, return original abstract
    if not v250 and not v300:
        print("⚠️ Rewrite parse failed; returning original abstract twice.")
        v250 = original_abstract
        v300 = original_abstract
    elif not v250:
        v250 = v300
    elif not v300:
        v300 = v250

    # Diagnostics
    wc250 = len(v250.split())
    wc300 = len(v300.split())
    print(f"📝 Word count (250 target): {wc250}")
    print(f"📝 Word count (300 target): {wc300}")

    return v250, v300