# scoring.py

from openai import OpenAI, RateLimitError
from config import OPENAI_API_KEY
import textstat
import ast
import json
import os
import hashlib

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Readability Scoring (unchanged)
# ---------------------------
def score_readability(text):
    return {
        "flesch_kincaid": textstat.flesch_kincaid_grade(text),
        "reading_ease": textstat.flesch_reading_ease(text),
        "word_count": textstat.lexicon_count(text)
    }

# ---------------------------
# Helpers
# ---------------------------
def clean_gpt_output(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )
    return content.strip()

def compute_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ---------------------------
# New rubric (Protocol v3.0)
# ---------------------------
def _rubric_v3():
    """
    14-item modified CONSORT-for-Abstracts rubric, max total = 25.
    Keys are canonical item names the model must return.
    """
    return {
        "Title": {"max": 1, "rule": "0 if title does not identify randomised trial; 1 if it does."},
        "Trial_Design": {"max": 2, "rule": "0 no design; 1 unit of randomisation stated; 2 unit + sample size calculation."},
        "Participants": {"max": 2, "rule": "0 none; 1 either eligibility OR setting; 2 both eligibility AND setting."},
        "Intervention": {"max": 2, "rule": "0 none; 1 intervention described; 2 intervention + comparator described."},
        "Objective": {"max": 1, "rule": "0 no objective; 1 objective stated."},
        "Method_Outcome": {"max": 2, "rule": "0 none; 1 primary outcome without timeframe; 2 primary outcome with timeframe."},
        "Randomisation_Allocation": {"max": 2, "rule": "0 neither; 1 either method OR allocation concealment; 2 both."},
        "Blinding": {"max": 3, "rule": "0 none; 1 one of assessor/patient/clinician; 2 two; 3 all three or explicitly triple/unblinded/open-label."},
        "Number_Randomised": {"max": 2, "rule": "0 none; 1 numbers per group; 2 numbers per group + recruitment period."},
        "Number_Analysed": {"max": 2, "rule": "0 none; 1 either per-group analysed OR ITT/per-protocol stated; 2 both."},
        "Result_Outcome": {"max": 3, "rule": "0 none; 1 group results only; 2 group results + effect size; 3 + precision (e.g., 95% CI)."},
        "Harms": {"max": 1, "rule": "0 no adverse events reporting; 1 adverse events reported."},
        "Trial_Registration": {"max": 1, "rule": "0 not stated; 1 register name + number provided."},
        "Funding": {"max": 1, "rule": "0 not reported; 1 source of funding reported."}
    }

_RUBRIC = _rubric_v3()
_RUBRIC_VERSION = "v3.0-2025-10-14"
_MAX_TOTAL = sum(spec["max"] for spec in _RUBRIC.values())  # 25

def _empty_result():
    return {
        "rubric_version": _RUBRIC_VERSION,
        "items": {k: {"score": 0, "evidence": "Not reported"} for k in _RUBRIC},
        "total_score": 0,
        "max_score": _MAX_TOTAL
    }

def _coerce_schema(d: dict) -> dict:
    """
    Coerce/validate model output to strict schema:
    {rubric_version:str, items:{key:{score:int,evidence:str}}, total_score:int, max_score:int}
    Missing items -> score 0, 'Not reported'.
    Scores clamped to [0, item.max].
    """
    result = _empty_result()

    # items
    items = d.get("items", {})
    if isinstance(items, dict):
        for k, spec in _RUBRIC.items():
            v = items.get(k, {})
            score = v.get("score", 0)
            try:
                score = int(score)
            except Exception:
                score = 0
            # clamp
            score = max(0, min(spec["max"], score))
            evidence = v.get("evidence", "Not reported")
            if not isinstance(evidence, str) or not evidence.strip():
                evidence = "Not reported"
            result["items"][k] = {"score": score, "evidence": evidence}

    # total score (recompute to be safe)
    total = sum(v["score"] for v in result["items"].values())
    result["total_score"] = int(total)
    result["max_score"] = _MAX_TOTAL
    result["rubric_version"] = str(d.get("rubric_version", _RUBRIC_VERSION))
    return result

# ---------------------------
# CONSORT-like Scoring (updated to 14 items with evidence)
# ---------------------------
def score_consort(text: str, return_model: bool = False):
    """
    Scores an abstract against the protocol v3.0 14-item rubric (max 25) and
    returns strict JSON with per-item evidence.

    Output shape:
    {
      "rubric_version": "v3.0-2025-10-14",
      "items": {
        "Title": {"score": int, "evidence": str},
        ...
      },
      "total_score": int,
      "max_score": 25
    }
    """

    # ---- Cache by hash + rubric version ----
    hash_id = compute_sha256(text + _RUBRIC_VERSION)
    cache_dir = "data/consort_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{hash_id}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if return_model:
            return cached["scores"], cached.get("model", "unknown")
        else:
            return cached["scores"]

    # ---- Build prompt from rubric ----
    rubric_lines = []
    for name, spec in _RUBRIC.items():
        rubric_lines.append(f'- {name} (0–{spec["max"]}): {spec["rule"]}')

    prompt = (
        "You are an expert clinical researcher evaluating an RCT abstract using a 14-item, graded rubric.\n"
        "Return ONLY a strict JSON object with this schema:\n"
        "{\n"
        '  "rubric_version": "v3.0-2025-10-14",\n'
        '  "items": {\n'
        '    "<ItemName>": {"score": <int>, "evidence": "<exact supporting phrase/sentence or Not reported>"}\n'
        "    // include ALL items listed below\n"
        "  },\n"
        '  "total_score": <int>,\n'
        '  "max_score": 25\n'
        "}\n\n"
        "Rules:\n"
        "- Scores must be integers in the allowed range for each item.\n"
        "- Evidence must be an exact phrase/sentence from the abstract that supports the score; if absent, use 'Not reported'.\n"
        "- Do NOT include any commentary, markdown, code fences, or explanations—JSON only.\n\n"
        "Rubric:\n"
        + "\n".join(rubric_lines)
        + "\n\n"
        "Abstract to score:\n"
        f"{text}\n\n"
        "Now return the JSON only."
    )

    def _try_parse(content: str):
        content = clean_gpt_output(content)
        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(content)
            except Exception:
                return None

    def _call_model(model_name: str):
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw = response.choices[0].message.content
        parsed = _try_parse(raw)
        if not isinstance(parsed, dict):
            # fallback empty result if unparsable
            coerced = _empty_result()
        else:
            coerced = _coerce_schema(parsed)
        return coerced

    # ---- Primary model: gpt-5 ----
    used_model = "gpt-5"
    try:
        result = _call_model(used_model)
    except RateLimitError:
        # ---- Fallback: gpt-4o ----
        used_model = "gpt-4o"
        result = _call_model(used_model)
    except Exception:
        used_model = "gpt-4o"
        result = _call_model(used_model)

    # ---- Cache ----
    to_cache = {"scores": result, "model": used_model}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(to_cache, f, indent=2)

    return (result, used_model) if return_model else result
