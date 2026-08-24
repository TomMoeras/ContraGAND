"""Symbolic post-generation checks.

Cheap mechanical validators that run after LLM generation/evaluation.
Catch deterministic failures the LLM evaluator may miss (formatting drift,
edit-distance outliers, referent deletion in non-swap strategies, etc.).
"""

from __future__ import annotations

import re

NON_SWAP_STRATEGIES = {
    "pronoun_swap", "pronoun_insertion", "context_modifier",
    "relational_insertion", "title_insertion", "sir_maam",
    "adjective", "appositive_dash",
}

POS_CHANGE_SUFFIXES = ("man", "woman", "boy", "girl", "lady", "gentleman")


def word_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance on whitespace-split tokens."""
    x = (a or "").split()
    y = (b or "").split()
    n, m = len(x), len(y)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if x[i - 1] == y[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def check_edit_distance(source: str, variant: str, hard_limit: int = 8, soft_limit: int = 5) -> tuple[bool, str]:
    d = word_edit_distance(source, variant)
    if d > hard_limit:
        return False, f"edit_distance={d} exceeds hard limit {hard_limit}"
    if d > soft_limit:
        return True, f"edit_distance={d} over soft limit {soft_limit}"
    return True, ""


def check_formatting_preserved(source: str, variant: str) -> tuple[bool, str]:
    """Flag when source has non-standard contraction spacing but variant normalises it."""
    if not source or not variant:
        return True, ""
    weird_patterns = [r"I'\s*m\b", r"you'\s*re\b", r"don'\s*t\b", r"won'\s*t\b", r"can'\s*t\b", r"I'\s*ve\b", r"I'\s*ll\b"]
    for pat in weird_patterns:
        matches_src = re.findall(pat, source)
        if not matches_src:
            continue
        pat_has_space = r"'\s+" in pat
        for m in matches_src:
            if m in variant:
                continue
            normalised = re.sub(r"'\s+", "'", m)
            if normalised in variant and normalised != m:
                return False, f"formatting drift: '{m}' -> '{normalised}'"
    return True, ""


_TITLE_PREFIXES = r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bSt)"


def count_sentence_boundaries(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(rf"{_TITLE_PREFIXES}[.!?]\s+[A-Z]", text))


def check_no_extra_sentence(source: str, variant: str) -> tuple[bool, str]:
    src_n = count_sentence_boundaries(source)
    var_n = count_sentence_boundaries(variant)
    if var_n > src_n:
        return False, f"variant has {var_n} sentence boundaries vs source's {src_n}"
    return True, ""


def check_variants_differ(masc: str, fem: str) -> tuple[bool, str]:
    if (masc or "").strip() == (fem or "").strip():
        return False, "masculine and feminine variants are identical"
    return True, ""


def check_variants_edited(source: str, masc: str, fem: str) -> tuple[bool, str]:
    if (masc or "").strip() == (source or "").strip() and (fem or "").strip() == (source or "").strip():
        return False, "both variants identical to the source"
    return True, ""


def check_referent_preserved(source: str, variant: str, referent: str, strategy: str) -> tuple[bool, str]:
    if strategy == "referent_swap":
        return True, ""
    if not referent:
        return True, ""
    pat = r"\b" + re.escape(referent) + r"s?\b"
    if re.search(pat, source, flags=re.IGNORECASE) and not re.search(pat, variant, flags=re.IGNORECASE):
        return False, f"referent '{referent}' was removed under non-swap strategy '{strategy}'"
    return True, ""


def check_no_pos_change_symbolic(source: str, variant: str, referent: str) -> tuple[bool, str]:
    """Flag suspected POS change: 'the X' noun -> 'the X man' where X is now adjective."""
    if not referent:
        return True, ""
    ref = re.escape(referent)
    src_has = re.search(rf"\b{ref}\b(?!\s+({'|'.join(POS_CHANGE_SUFFIXES)}))", source, flags=re.IGNORECASE)
    var_has_suffix = re.search(rf"\b{ref}\s+({'|'.join(POS_CHANGE_SUFFIXES)})\b", variant, flags=re.IGNORECASE)
    if src_has and var_has_suffix:
        return False, f"possible POS change: '{referent} {var_has_suffix.group(1)}' turns referent into adjective"
    return True, ""


def validate_row(row) -> dict:
    """Run all symbolic checks on a generated row.

    Returns a dict {check_name: (passed: bool, message: str)} plus
    an overall 'passed' boolean and a 'messages' list of failure messages.
    """
    source = row.get("EN_source_sentence", "") or ""
    masc = row.get("EN_masculine_sentence", "") or ""
    fem = row.get("EN_feminine_sentence", "") or ""
    referent = row.get("referent", "") or ""
    strategy = row.get("disambiguation_strategy") or row.get("eval_strategy_label") or ""

    results = {
        "edit_distance_masc": check_edit_distance(source, masc),
        "edit_distance_fem": check_edit_distance(source, fem),
        "formatting_masc": check_formatting_preserved(source, masc),
        "formatting_fem": check_formatting_preserved(source, fem),
        "no_extra_sentence_masc": check_no_extra_sentence(source, masc),
        "no_extra_sentence_fem": check_no_extra_sentence(source, fem),
        "variants_differ": check_variants_differ(masc, fem),
        "variants_edited": check_variants_edited(source, masc, fem),
        "referent_preserved_masc": check_referent_preserved(source, masc, referent, strategy),
        "referent_preserved_fem": check_referent_preserved(source, fem, referent, strategy),
        "no_pos_change_masc": check_no_pos_change_symbolic(source, masc, referent),
        "no_pos_change_fem": check_no_pos_change_symbolic(source, fem, referent),
    }
    passed = all(p for p, _ in results.values())
    messages = [f"{k}: {msg}" for k, (p, msg) in results.items() if not p or msg]
    return {"passed": passed, "messages": messages, "details": results}
