"""Symbolic pre-analysis of (sentence, referent) pairs.

Produces deterministic hints that route generation to the most natural strategy
when the data signals it clearly. The hints are rendered into the user message
of the generation prompt so the LLM can follow them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

NEUTRAL_PRONOUNS = {"they", "their", "them", "themselves", "theirs"}
FIRST_PERSON_MARKERS = ("i'm", "i am", "i've", "i'll", "i had", "i was", "as a ")

# Pairs whose gendered forms are connotatively asymmetric or where the
# masculine form is often unmarked / identical to the source. We refuse to
# emit a referent_swap hint for these - the LLM is told to use a different
# strategy or keep the masculine variant identical to the source.
FORBIDDEN_SWAP_PAIRS = {
    ("hero", "heroine"),
    ("master", "mistress"),
    ("confidant", "confidante"),
}


@dataclass
class Hint:
    recommended_strategy: Optional[str] = None
    masc_referent: Optional[str] = None
    fem_referent: Optional[str] = None
    note: Optional[str] = None
    signals: dict = field(default_factory=dict)

    def render(self) -> str:
        lines = []
        if self.recommended_strategy:
            lines.append(f"RECOMMENDED_STRATEGY: {self.recommended_strategy}")
        if self.masc_referent and self.fem_referent:
            lines.append(f"MASC_REFERENT: {self.masc_referent}")
            lines.append(f"FEM_REFERENT: {self.fem_referent}")
        if self.note:
            lines.append(f"NOTE: {self.note}")
        if not lines:
            return ""
        return "\n[HINTS]\n" + "\n".join(lines)


def parse_lexicon(lexicon_md: str) -> dict:
    """Parse lexical_gender_nouns.md into {neutral_or_masc: (masc, fem)}."""
    table: dict[str, tuple[str, str]] = {}
    for line in lexicon_md.splitlines():
        line = line.strip()
        if not line.startswith("|") or "Masculine" in line or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        masc_raw, fem_raw = cells[0], cells[1]
        neutral_raw = cells[2] if len(cells) >= 3 else ""

        if masc_raw in {"-", "-", ""} or fem_raw in {"-", "-", ""}:
            continue

        masc_forms = [m.strip().lower() for m in re.split(r"/", masc_raw)]
        fem_forms = [f.strip().lower() for f in re.split(r"/", fem_raw)]
        neutral_forms = []
        if neutral_raw and neutral_raw not in {"-", "-"}:
            neutral_forms = [n.strip().lower() for n in re.split(r"[,/]", neutral_raw)]

        primary_masc = masc_forms[0]
        primary_fem = fem_forms[0]
        for key in masc_forms + neutral_forms:
            if key and key not in table:
                table[key] = (primary_masc, primary_fem)
        for fem_form in fem_forms:
            if fem_form and fem_form not in table:
                table[fem_form] = (primary_masc, primary_fem)
    return table


def load_gendered_pairs_lookup(md_path: str = "data/resources/lexical_gender_nouns.md") -> dict:
    with open(md_path, "r") as f:
        return parse_lexicon(f.read())


def detect_referent_pair(referent: str, lookup: dict) -> Optional[tuple[str, str]]:
    key = referent.strip().lower()
    if key in lookup:
        return lookup[key]
    stripped = re.sub(r"[^a-z]", "", key)
    if stripped in lookup:
        return lookup[stripped]
    return None


def detect_addressee(sentence: str, referent: str) -> bool:
    if not isinstance(sentence, str):
        return False
    ref = re.escape(referent)
    patterns = [
        rf"^(Yes|Hey|Look|Listen|Oh|No|Well|Please),?\s+{ref}\b",
        rf",\s+{ref}\s*[.!?]\s*$",
        rf",\s+{ref},",
        rf"\b{ref},?\s+(you|you're|you've|you'll|you'd)\b",
    ]
    return any(re.search(p, sentence, flags=re.IGNORECASE) for p in patterns)


def detect_first_person_referent(sentence: str, referent: str) -> bool:
    if not isinstance(sentence, str):
        return False
    lower = sentence.lower()
    ref_lower = referent.lower()
    if ref_lower not in lower:
        return False
    for marker in FIRST_PERSON_MARKERS:
        idx_marker = lower.find(marker)
        if idx_marker < 0:
            continue
        idx_ref = lower.find(ref_lower)
        if idx_ref < 0:
            continue
        distance = abs(idx_ref - idx_marker - len(marker))
        if distance <= 50:
            return True
    return False


def detect_existing_neutral_pronoun(sentence: str, referent: str, window_tokens: int = 10) -> set:
    found = set()
    if not isinstance(sentence, str):
        return found
    tokens = re.findall(r"[\w']+", sentence.lower())
    ref_lower = referent.lower()
    ref_positions = [i for i, t in enumerate(tokens) if t == ref_lower]
    if not ref_positions:
        return found
    for pos in ref_positions:
        lo = max(0, pos - window_tokens)
        hi = min(len(tokens), pos + window_tokens + 1)
        for j in range(lo, hi):
            if tokens[j] in NEUTRAL_PRONOUNS:
                found.add(tokens[j])
    return found


def analyze(row, lookup: dict) -> Hint:
    """Return a Hint for this row based on the strongest symbolic signal.

    Priority:
      1. Referent has a gendered pair in the lexicon -> referent_swap.
      2. Referent appears as an addressee -> sir_maam.
      3. Source has a neutral pronoun (they/their/…) referring to the referent -> pronoun_swap.
      4. Source is first-person ('I'm a X') -> pronoun_swap with symmetric swap.
      5. Otherwise: no hint.
    """
    referent = row["referent"]
    sentence = row["EN_source_sentence"]

    signals = {}
    pair = detect_referent_pair(referent, lookup)
    if pair and pair in FORBIDDEN_SWAP_PAIRS:
        signals["has_pair"] = True
        signals["forbidden_pair"] = True
        pair = None
    if pair:
        signals["has_pair"] = True
        return Hint(
            recommended_strategy="referent_swap",
            masc_referent=pair[0],
            fem_referent=pair[1],
            signals=signals,
        )

    if detect_addressee(sentence, referent):
        signals["addressee"] = True
        return Hint(
            recommended_strategy="sir_maam",
            note=f"Referent '{referent}' is the addressee - use sir/ma'am as a vocative.",
            signals=signals,
        )

    existing_pronouns = detect_existing_neutral_pronoun(sentence, referent)
    if existing_pronouns:
        signals["existing_pronouns"] = sorted(existing_pronouns)
        targets = ", ".join(sorted(existing_pronouns))
        return Hint(
            recommended_strategy="pronoun_swap",
            note=f"Source has {targets} referring to '{referent}' - swap to he/she, his/her, him/her, himself/herself.",
            signals=signals,
        )

    if detect_first_person_referent(sentence, referent):
        signals["first_person"] = True
        return Hint(
            recommended_strategy="pronoun_swap",
            note=f"Referent '{referent}' is the first-person speaker - swap every I/me/my that refers to the speaker.",
            signals=signals,
        )

    return Hint(signals=signals)
