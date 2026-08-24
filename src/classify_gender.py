"""Haiku gender classification on the v16 test-set contrastive pairs.

Two conditions: zero_shot and few_shot. Each classifies every (sentence, referent)
triple (source + masculine + feminine) in the test split.

Usage:
    # Run classification under a condition
    python classify_gender.py classify \\
        --input data/test-00000-of-00001_with_variants.parquet \\
        --model claude-haiku-4-5-20251001 \\
        --condition zero_shot \\
        --output results/classify/test_haiku_zero_shot.parquet

    # Side-by-side metrics
    python classify_gender.py evaluate \\
        --zero-shot results/classify/test_haiku_zero_shot.parquet \\
        --few-shot  results/classify/test_haiku_few_shot.parquet \\
        --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

SYSTEM_TASK = """You classify the gender of a named REFERENT in an English sentence.

Output one of three labels:
- masculine  - the referent is unambiguously male
- feminine   - the referent is unambiguously female
- ambiguous  - the sentence contains no signal that reveals the referent's gender

Respond with ONLY a JSON object:
{"gender": "masculine|feminine|ambiguous", "confidence": N, "reasoning": "..."}

confidence is an integer 1-5 (5 = certain).
reasoning is one short sentence citing the textual evidence (the pronoun, the title, the gendered noun, etc.)."""


SYSTEM_TASK_IDENTIFY = """You read an English sentence, identify every role, occupation, or relational REFERENT in it (e.g. doctor, teacher, brother, neighbor, captain), and classify each one's gender.

For every referent you find, classify it as one of:
- masculine  - unambiguously male
- feminine   - unambiguously female
- ambiguous  - no signal in the sentence reveals this referent's gender

Use the surface form as it appears in the sentence (e.g. "doctor", not "the doctor"). List the referents in the order they appear.

Respond with ONLY a JSON object:
{"referents": [{"referent": "...", "gender": "masculine|feminine|ambiguous", "confidence": N, "reasoning": "..."}]}

confidence is an integer 1-5 (5 = certain).
reasoning is one short sentence per referent citing the textual evidence."""


OPEN_CONDITIONS = {"zero_shot_identify"}


def is_open_condition(condition: str) -> bool:
    return condition in OPEN_CONDITIONS


def find_target_in_open_response(data: dict, target_referent: str) -> dict:
    """Locate a referent entry in an open-task response.

    Returns the matching entry dict, or {} if not found / malformed.
    Match strategy: case-insensitive exact match on whitespace-trimmed surface
    form, with a substring fallback so 'the doctor' matches our annotated
    'doctor' (and vice versa).
    """
    if not isinstance(data, dict):
        return {}
    referents = data.get("referents")
    if not isinstance(referents, list):
        return {}
    target = (target_referent or "").lower().strip().rstrip(".,!?:;")
    if not target:
        return {}

    def _norm(s: str) -> str:
        return (s or "").lower().strip().rstrip(".,!?:;")

    for entry in referents:
        if isinstance(entry, dict) and _norm(entry.get("referent")) == target:
            return entry
    for entry in referents:
        if not isinstance(entry, dict):
            continue
        ref = _norm(entry.get("referent"))
        if not ref:
            continue
        if target in ref.split() or ref in target.split():
            return entry
        if target in ref or ref in target:
            return entry
    return {}


FEW_SHOT_EXAMPLES = """Examples:

Sentence: "The engineer said they would finish the project on time."
Referent: engineer
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "Only 'they' is used; no gendered marker."}

Sentence: "The engineer, Mr. Harris, said he would finish the project on time."
Referent: engineer
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'Mr. Harris' and 'he' both point to male."}

Sentence: "The engineer, Mrs. Harris, said she would finish the project on time."
Referent: engineer
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'Mrs. Harris' and 'she' both point to female."}

Sentence: "The captain led the crew through the storm with steady hands."
Referent: captain
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "No pronoun, title, or gendered noun."}

Sentence: "The waiter brought us fresh bread and filled our glasses."
Referent: waiter
Output: {"gender": "masculine", "confidence": 4, "reasoning": "'Waiter' is the gendered masculine form of this role."}

Sentence: "The waitress brought us fresh bread and filled our glasses."
Referent: waitress
Output: {"gender": "feminine", "confidence": 4, "reasoning": "'Waitress' is the gendered feminine form of this role."}

Sentence: "Yes, officer, ma'am, I will move along."
Referent: officer
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'ma'am' is a feminine address for the officer."}

Sentence: "The nurse, his brother, checked my blood pressure twice."
Referent: nurse
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'his brother' identifies the nurse as male."}

Sentence: "A concerned bystander took photos at the scene."
Referent: bystander
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "No pronoun, title, or gendered noun."}"""


FEW_SHOT_EXAMPLE_SENTENCES = [
    "The engineer said they would finish the project on time.",
    "The engineer, Mr. Harris, said he would finish the project on time.",
    "The engineer, Mrs. Harris, said she would finish the project on time.",
    "The captain led the crew through the storm with steady hands.",
    "The waiter brought us fresh bread and filled our glasses.",
    "The waitress brought us fresh bread and filled our glasses.",
    "Yes, officer, ma'am, I will move along.",
    "The nurse, his brother, checked my blood pressure twice.",
    "A concerned bystander took photos at the scene.",
]


FEW_SHOT_FULL_EXAMPLES = """Examples:

Sentence: "The architect said they would send the revised blueprints tomorrow."
Referent: architect
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "Only 'they' is used as pronoun; no gendered marker."}

Sentence: "A quiet bookkeeper entered the office without a word."
Referent: bookkeeper
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "No pronoun, title, or gendered noun."}

Sentence: "The tour guide waved for everyone to follow along the path."
Referent: tour guide
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "No gendered cues present."}

Sentence: "The referee called a penalty and he explained the infraction."
Referent: referee
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'he' refers to the referee (pronoun swap from a neutral form)."}

Sentence: "The designer said she would submit the portfolio next week."
Referent: designer
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'she' refers to the designer (pronoun swap)."}

Sentence: "The violinist tuned his instrument before the rehearsal began."
Referent: violinist
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'his' is inserted near 'violinist' and refers to it."}

Sentence: "The violinist tuned her instrument before the rehearsal began."
Referent: violinist
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'her' is inserted near 'violinist' and refers to it."}

Sentence: "The actor accepted the award with a short, warm speech."
Referent: actor
Output: {"gender": "masculine", "confidence": 4, "reasoning": "'actor' is a gendered masculine form of this role."}

Sentence: "The actress accepted the award with a short, warm speech."
Referent: actress
Output: {"gender": "feminine", "confidence": 4, "reasoning": "'actress' is a gendered feminine form of this role."}

Sentence: "That man, the baker, kneaded the dough with practised ease."
Referent: baker
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'that man' is an appositive context modifier identifying the baker as male."}

Sentence: "That woman, the baker, kneaded the dough with practised ease."
Referent: baker
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'that woman' is an appositive context modifier identifying the baker as female."}

Sentence: "The librarian, Mr. Thompson, shelved the returned books."
Referent: librarian
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'Mr. Thompson' is a masculine title+name appositive for the librarian."}

Sentence: "The librarian, Mrs. Thompson, shelved the returned books."
Referent: librarian
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'Mrs. Thompson' is a feminine title+name appositive for the librarian."}

Sentence: "Yes, coach, sir, I will hustle next time."
Referent: coach
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'sir' is a masculine vocative addressing the coach."}

Sentence: "Yes, coach, ma'am, I will hustle next time."
Referent: coach
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'ma'am' is a feminine vocative addressing the coach."}

Sentence: "I spoke with the male paramedic who treated the injury."
Referent: paramedic
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'male paramedic' is an explicit adjective."}

Sentence: "I spoke with the female paramedic who treated the injury."
Referent: paramedic
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'female paramedic' is an explicit adjective."}

Sentence: "The sculptor - a man - stood proudly next to the marble bust."
Referent: sculptor
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'- a man -' is a masculine dash appositive for the sculptor."}

Sentence: "The sculptor - a woman - stood proudly next to the marble bust."
Referent: sculptor
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'- a woman -' is a feminine dash appositive for the sculptor."}

Sentence: "The gardener, his uncle, pruned the hedge on Saturday morning."
Referent: gardener
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'his uncle' identifies the gardener as male (relational insertion)."}

Sentence: "The gardener, her aunt, pruned the hedge on Saturday morning."
Referent: gardener
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'her aunt' identifies the gardener as female (relational insertion)."}"""


FEW_SHOT_FULL_EXAMPLE_SENTENCES = [
    "The architect said they would send the revised blueprints tomorrow.",
    "A quiet bookkeeper entered the office without a word.",
    "The tour guide waved for everyone to follow along the path.",
    "The referee called a penalty and he explained the infraction.",
    "The designer said she would submit the portfolio next week.",
    "The violinist tuned his instrument before the rehearsal began.",
    "The violinist tuned her instrument before the rehearsal began.",
    "The actor accepted the award with a short, warm speech.",
    "The actress accepted the award with a short, warm speech.",
    "That man, the baker, kneaded the dough with practised ease.",
    "That woman, the baker, kneaded the dough with practised ease.",
    "The librarian, Mr. Thompson, shelved the returned books.",
    "The librarian, Mrs. Thompson, shelved the returned books.",
    "Yes, coach, sir, I will hustle next time.",
    "Yes, coach, ma'am, I will hustle next time.",
    "I spoke with the male paramedic who treated the injury.",
    "I spoke with the female paramedic who treated the injury.",
    "The sculptor - a man - stood proudly next to the marble bust.",
    "The sculptor - a woman - stood proudly next to the marble bust.",
    "The gardener, his uncle, pruned the hedge on Saturday morning.",
    "The gardener, her aunt, pruned the hedge on Saturday morning.",
]


FEW_SHOT_V3_EXAMPLES = FEW_SHOT_FULL_EXAMPLES + """

Sentence: "Right away, captain, sir! The flare is ready."
Referent: captain
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'sir' addresses the captain as male."}

Sentence: "Step back, officer, sir - I will comply."
Referent: officer
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'sir' addresses the officer as male."}

Sentence: "Understood, judge, sir, I withdraw my objection."
Referent: judge
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'sir' addresses the judge as male."}

Sentence: "When you address a role by its masculine or feminine gendered form - for example, waiter, waitress, actor, actress, hero, heroine, fireman, firewoman, salesman, saleswoman, chairman, chairwoman - the gender is unambiguous."
Referent: waiter
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'waiter' is the masculine form."}

Sentence: "The actor gave a stirring monologue from centre stage."
Referent: actor
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'actor' is the masculine gendered form of this role."}

Sentence: "The heiress thanked the crowd for their support."
Referent: heiress
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'heiress' is the feminine gendered form."}"""


FEW_SHOT_V3_EXAMPLE_SENTENCES = FEW_SHOT_FULL_EXAMPLE_SENTENCES + [
    "Right away, captain, sir! The flare is ready.",
    "Step back, officer, sir - I will comply.",
    "Understood, judge, sir, I withdraw my objection.",
    "When you address a role by its masculine or feminine gendered form - for example, waiter, waitress, actor, actress, hero, heroine, fireman, firewoman, salesman, saleswoman, chairman, chairwoman - the gender is unambiguous.",
    "The actor gave a stirring monologue from centre stage.",
    "The heiress thanked the crowd for their support.",
]


FEW_SHOT_V4_EXAMPLES = FEW_SHOT_V3_EXAMPLES + """

Sentence: "The officer asked the man to step aside while they checked his ID."
Referent: officer
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "'the man' and 'his' refer to the person being checked, not to the officer. No gendered cue points to the officer."}

Sentence: "My sister's accountant reviewed the tax filing."
Referent: accountant
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "'sister' refers to the speaker's sister, not the accountant. No gendered cue points to the accountant."}

Sentence: "The doctor greeted the nurse as she walked into the ward."
Referent: doctor
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "'she' refers to the nurse, not the doctor. No gendered cue points to the doctor."}

Sentence: "He told his coach that he would train harder next season."
Referent: coach
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "'he' and 'his' refer to the speaker, not the coach. No gendered cue points to the coach."}"""


FEW_SHOT_V4_EXAMPLE_SENTENCES = FEW_SHOT_V3_EXAMPLE_SENTENCES + [
    "The officer asked the man to step aside while they checked his ID.",
    "My sister's accountant reviewed the tax filing.",
    "The doctor greeted the nurse as she walked into the ward.",
    "He told his coach that he would train harder next season.",
]


FEW_SHOT_V5_PREAMBLE = """Key rule: a gendered cue reveals the referent's gender only when it refers to the REFERENT. A gendered pronoun or noun that refers to someone else in the sentence does NOT disambiguate the referent. Read carefully and ask yourself: who is the cue about?

Common strategies that DO disambiguate the referent:
- PRONOUN SWAP/INSERTION: he/she/his/her/him in a position clearly referring to the referent.
- REFERENT SWAP: the referent word is itself a gendered form (waiter/waitress, actor/actress, fireman/firewoman, heir/heiress, etc.).
- CONTEXT MODIFIER: an upstream 'that man'/'that woman'/'a man'/'a woman' used as an appositive to the referent.
- TITLE INSERTION: Mr. / Mrs. / Ms. + name immediately next to the referent.
- SIR / MA'AM: used as a direct-address vocative for the referent (e.g. 'Yes, coach, sir').
- ADJECTIVE: 'male X' / 'female X' adjective immediately before the referent.
- APPOSITIVE DASH: '- a man -' / '- a woman -' dash appositive for the referent.
- RELATIONAL INSERTION: 'his brother' / 'her sister' / 'his uncle' / 'her aunt' appositive identifying the referent as a relative.

If none of these apply, the referent is ambiguous.
"""


FEW_SHOT_V5_EXAMPLES = FEW_SHOT_V5_PREAMBLE + "\n" + FEW_SHOT_V4_EXAMPLES


FEW_SHOT_V5_EXAMPLE_SENTENCES = FEW_SHOT_V4_EXAMPLE_SENTENCES


FEW_SHOT_V6_EXAMPLES = """Examples:

Sentence: "The architect said they would send the revised blueprints tomorrow."
Referent: architect
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "Only 'they' is used as pronoun; no gendered marker."}

Sentence: "A quiet bookkeeper entered the office without a word."
Referent: bookkeeper
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "No pronoun, title, or gendered noun."}

Sentence: "The tour guide waved for everyone to follow along the path."
Referent: tour guide
Output: {"gender": "ambiguous", "confidence": 5, "reasoning": "No gendered cues present."}

Sentence: "The actor accepted the award with a short, warm speech."
Referent: actor
Output: {"gender": "masculine", "confidence": 4, "reasoning": "'actor' is a gendered masculine form of this role."}

Sentence: "The actress accepted the award with a short, warm speech."
Referent: actress
Output: {"gender": "feminine", "confidence": 4, "reasoning": "'actress' is a gendered feminine form of this role."}

Sentence: "I spoke with the male paramedic who treated the injury."
Referent: paramedic
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'male paramedic' is an explicit adjective."}

Sentence: "I spoke with the female paramedic who treated the injury."
Referent: paramedic
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'female paramedic' is an explicit adjective."}

Sentence: "The sculptor - a man - stood proudly next to the marble bust."
Referent: sculptor
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'- a man -' is a masculine dash appositive for the sculptor."}

Sentence: "The sculptor - a woman - stood proudly next to the marble bust."
Referent: sculptor
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'- a woman -' is a feminine dash appositive for the sculptor."}

Sentence: "That man, the baker, kneaded the dough with practised ease."
Referent: baker
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'that man' is an appositive context modifier identifying the baker as male."}

Sentence: "That woman, the baker, kneaded the dough with practised ease."
Referent: baker
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'that woman' is an appositive context modifier identifying the baker as female."}

Sentence: "The referee called a penalty and he explained the infraction."
Referent: referee
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'he' refers to the referee (pronoun swap from a neutral form)."}

Sentence: "The designer said she would submit the portfolio next week."
Referent: designer
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'she' refers to the designer (pronoun swap)."}

Sentence: "The violinist tuned his instrument before the rehearsal began."
Referent: violinist
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'his' is inserted near 'violinist' and refers to it."}

Sentence: "The violinist tuned her instrument before the rehearsal began."
Referent: violinist
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'her' is inserted near 'violinist' and refers to it."}

Sentence: "The librarian, Mr. Thompson, shelved the returned books."
Referent: librarian
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'Mr. Thompson' is a masculine title+name appositive for the librarian."}

Sentence: "The librarian, Mrs. Thompson, shelved the returned books."
Referent: librarian
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'Mrs. Thompson' is a feminine title+name appositive for the librarian."}

Sentence: "The gardener, his uncle, pruned the hedge on Saturday morning."
Referent: gardener
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'his uncle' identifies the gardener as male (relational insertion)."}

Sentence: "The gardener, her aunt, pruned the hedge on Saturday morning."
Referent: gardener
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'her aunt' identifies the gardener as female (relational insertion)."}

Sentence: "Yes, coach, sir, I will hustle next time."
Referent: coach
Output: {"gender": "masculine", "confidence": 5, "reasoning": "'sir' is a masculine vocative addressing the coach."}

Sentence: "Yes, coach, ma'am, I will hustle next time."
Referent: coach
Output: {"gender": "feminine", "confidence": 5, "reasoning": "'ma'am' is a feminine vocative addressing the coach."}"""


FEW_SHOT_V6_EXAMPLE_SENTENCES = FEW_SHOT_FULL_EXAMPLE_SENTENCES


FEW_SHOT_V7_PREAMBLE = """Important: The question is always about the REFERENT. A gendered cue counts only when it clearly refers to the referent. If a pronoun, title, or gendered noun refers to a DIFFERENT person in the sentence, it does NOT disambiguate the referent."""


FEW_SHOT_V7_EXAMPLES = FEW_SHOT_V7_PREAMBLE + "\n\n" + FEW_SHOT_FULL_EXAMPLES


FEW_SHOT_V7_EXAMPLE_SENTENCES = FEW_SHOT_FULL_EXAMPLE_SENTENCES


def build_system_prompt(condition: str) -> str:
    if condition == "zero_shot":
        return SYSTEM_TASK
    if condition == "few_shot":
        return SYSTEM_TASK + "\n\n" + FEW_SHOT_EXAMPLES
    if condition == "few_shot_full":
        return SYSTEM_TASK + "\n\n" + FEW_SHOT_FULL_EXAMPLES
    if condition == "few_shot_v3":
        return SYSTEM_TASK + "\n\n" + FEW_SHOT_V3_EXAMPLES
    if condition == "few_shot_v4":
        return SYSTEM_TASK + "\n\n" + FEW_SHOT_V4_EXAMPLES
    if condition == "few_shot_v5":
        return SYSTEM_TASK + "\n\n" + FEW_SHOT_V5_EXAMPLES
    if condition == "few_shot_v6":
        return SYSTEM_TASK + "\n\n" + FEW_SHOT_V6_EXAMPLES
    if condition == "few_shot_v7":
        return SYSTEM_TASK + "\n\n" + FEW_SHOT_V7_EXAMPLES
    if condition == "zero_shot_identify":
        return SYSTEM_TASK_IDENTIFY
    raise ValueError(f"Unknown condition: {condition}")


def example_sentences_for(condition: str):
    if condition == "few_shot":
        return FEW_SHOT_EXAMPLE_SENTENCES
    if condition == "few_shot_full":
        return FEW_SHOT_FULL_EXAMPLE_SENTENCES
    if condition == "few_shot_v3":
        return FEW_SHOT_V3_EXAMPLE_SENTENCES
    if condition == "few_shot_v4":
        return FEW_SHOT_V4_EXAMPLE_SENTENCES
    if condition == "few_shot_v5":
        return FEW_SHOT_V5_EXAMPLE_SENTENCES
    if condition == "few_shot_v6":
        return FEW_SHOT_V6_EXAMPLE_SENTENCES
    if condition == "few_shot_v7":
        return FEW_SHOT_V7_EXAMPLE_SENTENCES
    return []


def to_long_form(df: pd.DataFrame) -> pd.DataFrame:
    """Expand each test row into three classification examples."""
    records = []
    for idx, row in df.iterrows():
        row_id = idx
        src_referent = row["referent"]
        masc_referent = row.get("masc_referent") if pd.notna(row.get("masc_referent")) else src_referent
        fem_referent = row.get("fem_referent") if pd.notna(row.get("fem_referent")) else src_referent

        records.append({
            "row_id": row_id,
            "variant": "source",
            "sentence": row["EN_source_sentence"],
            "referent": src_referent,
            "expected_label": "ambiguous",
        })
        if pd.notna(row.get("EN_masculine_sentence")):
            records.append({
                "row_id": row_id,
                "variant": "masculine",
                "sentence": row["EN_masculine_sentence"],
                "referent": masc_referent,
                "expected_label": "masculine",
            })
        if pd.notna(row.get("EN_feminine_sentence")):
            records.append({
                "row_id": row_id,
                "variant": "feminine",
                "sentence": row["EN_feminine_sentence"],
                "referent": fem_referent,
                "expected_label": "feminine",
            })
    return pd.DataFrame(records)


def parse_json_response(text: str) -> dict:
    """Parse the LAST top-level balanced JSON object that loads in text.

    Robust to chain-of-thought reasoning models that emit prose plus possibly
    intermediate JSON drafts before the final answer. For non-thinking models
    there's only one block, so "last" = "first".
    """
    if text is None:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    last_valid: dict = {}
    depth = 0
    start_idx = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx >= 0:
                    candidate = text[start_idx:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            last_valid = parsed
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
    return last_valid


def create_classify_requests(long_df: pd.DataFrame, system_prompt: str, model: str):
    requests = []
    for idx, row in long_df.iterrows():
        user_msg = f"Sentence: {row['sentence']}\nReferent: {row['referent']}"
        requests.append({
            "custom_id": f"cls_{idx}",
            "params": {
                "model": model,
                "max_tokens": 200,
                "system": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": user_msg}],
            },
        })
    return requests


def cmd_classify(args):
    import anthropic
    from generate_batch import collect_results, submit_batch, wait_for_batch
    client = anthropic.Anthropic()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    long_df = to_long_form(df)
    print(f"Expanded to {len(long_df)} classification examples ({args.condition}).")

    hygiene_check(long_df, args.condition)

    system_prompt = build_system_prompt(args.condition)
    requests = create_classify_requests(long_df, system_prompt, args.model)

    batch = submit_batch(client, requests)
    batch = wait_for_batch(client, batch.id)
    results = collect_results(client, batch.id)

    predictions = []
    confidences = []
    reasonings = []
    raw = []
    for idx in long_df.index:
        cid = f"cls_{idx}"
        if cid not in results:
            predictions.append(None)
            confidences.append(None)
            reasonings.append(None)
            raw.append(None)
            continue
        result = results[cid]
        if result.result.type != "succeeded":
            predictions.append("parse_error")
            confidences.append(None)
            reasonings.append(None)
            raw.append(None)
            continue
        text = result.result.message.content[0].text
        raw.append(text)
        data = parse_json_response(text)
        gender = data.get("gender")
        if gender in {"masculine", "feminine", "ambiguous"}:
            predictions.append(gender)
        else:
            predictions.append("parse_error")
        conf = data.get("confidence")
        if isinstance(conf, (int, float)):
            confidences.append(int(conf))
        else:
            confidences.append(None)
        reasonings.append(data.get("reasoning"))

    long_df = long_df.copy()
    long_df["condition"] = args.condition
    long_df["predicted_label"] = predictions
    long_df["confidence"] = confidences
    long_df["reasoning"] = reasonings
    long_df["correct"] = long_df["predicted_label"] == long_df["expected_label"]
    long_df["raw_response"] = raw

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    long_df.to_parquet(args.output, index=False)
    long_df.to_csv(args.output.replace(".parquet", ".csv"), index=False)

    total = len(long_df)
    correct = int(long_df["correct"].sum())
    parse_err = int((long_df["predicted_label"] == "parse_error").sum())
    print(f"\nClassified {total} examples. Correct: {correct}/{total} ({correct/total:.1%})")
    print(f"Parse errors: {parse_err}")
    print(f"Saved to {args.output}")


def hygiene_check(long_df: pd.DataFrame, condition: str):
    examples = example_sentences_for(condition)
    if not examples:
        return
    offenders = []
    test_sentences = set(long_df["sentence"])
    for fs in examples:
        if fs in test_sentences:
            offenders.append(fs)
    for fs in examples:
        for t in test_sentences:
            if fs and t and fs != t and (fs in t or t in fs):
                offenders.append(f"SUBSTRING: {fs!r} <-> {t!r}")
                break
    if offenders:
        print("HYGIENE CHECK FAILED: few-shot examples overlap with test sentences:")
        for o in offenders[:5]:
            print(f"  {o}")
        raise SystemExit(1)


def build_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    labels = ["masculine", "feminine", "ambiguous", "parse_error", "not_found"]
    mat = pd.DataFrame(0, index=labels[:3], columns=labels, dtype=int)
    for _, row in df.iterrows():
        expected = row["expected_label"]
        predicted = row["predicted_label"]
        if expected in mat.index and predicted in mat.columns:
            mat.at[expected, predicted] += 1
    return mat


def per_class_prf(df: pd.DataFrame) -> pd.DataFrame:
    labels = ["masculine", "feminine", "ambiguous"]
    rows = []
    for label in labels:
        tp = int(((df["predicted_label"] == label) & (df["expected_label"] == label)).sum())
        fp = int(((df["predicted_label"] == label) & (df["expected_label"] != label)).sum())
        fn = int(((df["predicted_label"] != label) & (df["expected_label"] == label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"label": label, "tp": tp, "fp": fp, "fn": fn,
                      "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)})
    out = pd.DataFrame(rows)
    macro = {
        "label": "macro",
        "tp": out["tp"].sum(), "fp": out["fp"].sum(), "fn": out["fn"].sum(),
        "precision": round(out["precision"].mean(), 4),
        "recall": round(out["recall"].mean(), 4),
        "f1": round(out["f1"].mean(), 4),
    }
    out = pd.concat([out, pd.DataFrame([macro])], ignore_index=True)
    return out


def breakdown_by_strategy(classified: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    joined = classified.merge(
        test_df[["eval_strategy_label"]].rename_axis("row_id").reset_index(),
        on="row_id",
        how="left",
    )
    masc_fem = joined[joined["variant"].isin(["masculine", "feminine"])]
    grouped = masc_fem.groupby(["eval_strategy_label", "variant"])["correct"]
    rows = []
    for (strat, variant), vals in grouped:
        rows.append({
            "strategy": strat,
            "variant": variant,
            "n": len(vals),
            "accuracy": round(float(vals.mean()), 4),
        })
    src = joined[joined["variant"] == "source"]
    if len(src) > 0:
        rows.append({
            "strategy": "(source rows - ambiguous)",
            "variant": "source",
            "n": len(src),
            "accuracy": round(float(src["correct"].mean()), 4),
        })
    return pd.DataFrame(rows).sort_values(["strategy", "variant"]).reset_index(drop=True)


def breakdown_by_flag(classified: pd.DataFrame, test_df: pd.DataFrame, flag: str) -> pd.DataFrame:
    if flag not in test_df.columns:
        return pd.DataFrame()
    joined = classified.merge(
        test_df[[flag]].rename_axis("row_id").reset_index(),
        on="row_id",
        how="left",
    )
    rows = []
    for val, grp in joined.groupby(flag):
        rows.append({
            flag: bool(val) if not pd.isna(val) else None,
            "n": len(grp),
            "accuracy": round(float(grp["correct"].mean()), 4),
        })
    return pd.DataFrame(rows)


def breakdown_by_acceptance(classified: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    if "accepted" not in test_df.columns:
        return pd.DataFrame()
    joined = classified.merge(
        test_df[["accepted"]].rename_axis("row_id").reset_index(),
        on="row_id",
        how="left",
    )
    rows = []
    for val, grp in joined.groupby("accepted"):
        rows.append({
            "accepted": bool(val) if not pd.isna(val) else None,
            "n": len(grp),
            "accuracy": round(float(grp["correct"].mean()), 4),
        })
    return pd.DataFrame(rows)


def confidence_calibration(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for correct_val in [True, False]:
        sub = df[df["correct"] == correct_val]
        rows.append({
            "correct": correct_val,
            "n": len(sub),
            "mean_confidence": round(float(sub["confidence"].dropna().mean()), 3) if len(sub) else 0.0,
        })
    return pd.DataFrame(rows)


def top_referents(classified: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    counts = classified["referent"].value_counts()
    top = counts.head(n).index.tolist()
    rows = []
    for ref in top:
        sub = classified[classified["referent"] == ref]
        rows.append({
            "referent": ref,
            "n": len(sub),
            "accuracy": round(float(sub["correct"].mean()), 4),
        })
    return pd.DataFrame(rows)


def ambiguity_bias_table(classified: pd.DataFrame) -> pd.DataFrame:
    """On source rows where the model picked masculine or feminine with conf >= 4,
    show per-referent male vs female guessing distribution."""
    src = classified[classified["variant"] == "source"]
    confident_wrong = src[(src["predicted_label"].isin(["masculine", "feminine"])) &
                         (src["confidence"].fillna(0) >= 4)]
    rows = []
    for ref, grp in confident_wrong.groupby("referent"):
        male = int((grp["predicted_label"] == "masculine").sum())
        female = int((grp["predicted_label"] == "feminine").sum())
        if male + female == 0:
            continue
        rows.append({
            "referent": ref,
            "n_confident_guesses": male + female,
            "guessed_masculine": male,
            "guessed_feminine": female,
            "male_share": round(male / (male + female), 3),
        })
    if not rows:
        return pd.DataFrame(columns=["referent", "n_confident_guesses",
                                       "guessed_masculine", "guessed_feminine",
                                       "male_share"])
    out = pd.DataFrame(rows).sort_values("n_confident_guesses", ascending=False).reset_index(drop=True)
    return out


def cmd_evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    test_df = pd.read_parquet(args.test_input)
    print(f"Loaded test input: {args.test_input}  ({len(test_df)} rows)")

    conditions = {}
    if args.zero_shot:
        conditions["zero_shot"] = pd.read_parquet(args.zero_shot)
    if args.few_shot:
        conditions["few_shot"] = pd.read_parquet(args.few_shot)
    if args.few_shot_full:
        conditions["few_shot_full"] = pd.read_parquet(args.few_shot_full)
    if getattr(args, "few_shot_v3", None):
        conditions["few_shot_v3"] = pd.read_parquet(args.few_shot_v3)
    if getattr(args, "few_shot_v4", None):
        conditions["few_shot_v4"] = pd.read_parquet(args.few_shot_v4)
    if getattr(args, "few_shot_v5", None):
        conditions["few_shot_v5"] = pd.read_parquet(args.few_shot_v5)
    if getattr(args, "few_shot_v6", None):
        conditions["few_shot_v6"] = pd.read_parquet(args.few_shot_v6)
    if getattr(args, "few_shot_v7", None):
        conditions["few_shot_v7"] = pd.read_parquet(args.few_shot_v7)
    if getattr(args, "zero_shot_identify", None):
        conditions["zero_shot_identify"] = pd.read_parquet(args.zero_shot_identify)

    report_lines = ["# Haiku gender classification - report\n"]

    per_condition_metrics = {}

    for name, df in conditions.items():
        report_lines.append(f"## Condition: {name}\n")
        total = len(df)
        correct = int(df["correct"].sum())
        overall_acc = correct / total if total else 0.0
        parse_err = int((df["predicted_label"] == "parse_error").sum())
        report_lines.append(f"- Examples: {total}")
        report_lines.append(f"- Overall accuracy: **{overall_acc:.3f}** ({correct}/{total})")
        report_lines.append(f"- Parse errors: {parse_err}")
        report_lines.append("")

        cm = build_confusion_matrix(df)
        cm_path = os.path.join(args.output_dir, f"confusion_{name}.csv")
        cm.to_csv(cm_path)
        report_lines.append(f"Confusion matrix (rows=expected, cols=predicted):\n```\n{cm}\n```\n")

        prf = per_class_prf(df)
        prf_path = os.path.join(args.output_dir, f"prf_{name}.csv")
        prf.to_csv(prf_path, index=False)
        report_lines.append(f"Per-class precision/recall/F1:\n```\n{prf.to_string(index=False)}\n```\n")

        strat = breakdown_by_strategy(df, test_df)
        strat.to_csv(os.path.join(args.output_dir, f"by_strategy_{name}.csv"), index=False)
        report_lines.append(f"Breakdown by strategy (masc/fem variants):\n```\n{strat.to_string(index=False)}\n```\n")

        acc = breakdown_by_acceptance(df, test_df)
        if len(acc) > 0:
            acc.to_csv(os.path.join(args.output_dir, f"by_acceptance_{name}.csv"), index=False)
            report_lines.append(f"Breakdown by pipeline acceptance:\n```\n{acc.to_string(index=False)}\n```\n")

        for flag in ["filter_any", "filter_multi_sentence", "filter_pre_gendered"]:
            fb = breakdown_by_flag(df, test_df, flag)
            if len(fb) > 0:
                fb.to_csv(os.path.join(args.output_dir, f"by_{flag}_{name}.csv"), index=False)
                report_lines.append(f"Breakdown by {flag}:\n```\n{fb.to_string(index=False)}\n```\n")

        conf_cal = confidence_calibration(df)
        conf_cal.to_csv(os.path.join(args.output_dir, f"confidence_calibration_{name}.csv"), index=False)
        report_lines.append(f"Confidence calibration:\n```\n{conf_cal.to_string(index=False)}\n```\n")

        refs = top_referents(df, n=20)
        refs.to_csv(os.path.join(args.output_dir, f"top_referents_{name}.csv"), index=False)
        report_lines.append(f"Top 20 referents by frequency:\n```\n{refs.to_string(index=False)}\n```\n")

        bias = ambiguity_bias_table(df)
        bias.to_csv(os.path.join(args.output_dir, f"ambiguity_bias_{name}.csv"), index=False)
        if len(bias) > 0:
            report_lines.append(f"Stereotypical bias on ambiguous sources (confidence ≥ 4):\n```\n{bias.head(15).to_string(index=False)}\n```\n")

        misc = df[df["correct"] == False].sort_values("confidence")
        low = misc.head(50)
        high = misc.sort_values("confidence", ascending=False).head(50)
        low.to_csv(os.path.join(args.output_dir, f"misclassified_low_conf_{name}.csv"), index=False)
        high.to_csv(os.path.join(args.output_dir, f"misclassified_high_conf_{name}.csv"), index=False)

        per_condition_metrics[name] = {
            "overall": overall_acc,
            "macro_f1": float(prf[prf["label"] == "macro"]["f1"].iloc[0]),
            "ambiguity_recall": float(prf[prf["label"] == "ambiguous"]["recall"].iloc[0]),
            "masculine_recall": float(prf[prf["label"] == "masculine"]["recall"].iloc[0]),
            "feminine_recall": float(prf[prf["label"] == "feminine"]["recall"].iloc[0]),
        }

    if len(conditions) >= 2:
        report_lines.append("## All-conditions headline summary\n")
        summary_rows = []
        metrics = ["overall", "macro_f1", "ambiguity_recall", "masculine_recall", "feminine_recall"]
        for metric in metrics:
            row = {"metric": metric}
            for name in conditions:
                row[name] = round(per_condition_metrics[name][metric], 4)
            summary_rows.append(row)
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(os.path.join(args.output_dir, "all_conditions_summary.csv"), index=False)
        report_lines.append(f"```\n{summary.to_string(index=False)}\n```\n")

        names = list(conditions.keys())
        pairs = []
        if "zero_shot" in names:
            for n in names:
                if n != "zero_shot":
                    pairs.append(("zero_shot", n))
        else:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    pairs.append((names[i], names[j]))

        for a, b in pairs:
            report_lines.append(f"## {a}  vs  {b}\n")
            a_df = conditions[a].set_index(["row_id", "variant"])
            b_df = conditions[b].set_index(["row_id", "variant"])
            joined = a_df.join(b_df, lsuffix=f"_{a}", rsuffix=f"_{b}", how="inner")

            fixed = joined[(joined[f"correct_{a}"] == False) & (joined[f"correct_{b}"] == True)]
            broken = joined[(joined[f"correct_{a}"] == True) & (joined[f"correct_{b}"] == False)]
            fixed.to_csv(os.path.join(args.output_dir, f"fixed_by_{b}_vs_{a}.csv"))
            broken.to_csv(os.path.join(args.output_dir, f"broken_by_{b}_vs_{a}.csv"))
            report_lines.append(f"- Fixed by {b} (wrong in {a}): {len(fixed)}")
            report_lines.append(f"- Broken by {b} (correct in {a}): {len(broken)}")
            report_lines.append(f"- Net gain ({b} − {a}): {len(fixed) - len(broken)} examples")

            agree = int((joined[f"predicted_label_{a}"] == joined[f"predicted_label_{b}"]).sum())
            report_lines.append(f"- Agreement rate: {agree}/{len(joined)} = {agree / len(joined):.3f}")

            cm_a = build_confusion_matrix(conditions[a])
            cm_b = build_confusion_matrix(conditions[b])
            diff = cm_b.astype(int) - cm_a.astype(int)
            diff.to_csv(os.path.join(args.output_dir, f"confusion_diff_{b}_minus_{a}.csv"))
            report_lines.append(f"\nConfusion-matrix diff ({b} − {a}):\n```\n{diff}\n```\n")

    report_path = os.path.join(args.output_dir, "haiku_test_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote {report_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("classify", help="Classify a condition's examples")
    c.add_argument("--input", required=True, help="test parquet with variants")
    c.add_argument("--model", default="claude-haiku-4-5-20251001")
    c.add_argument("--condition", required=True,
                   choices=["zero_shot", "few_shot", "few_shot_full",
                             "few_shot_v3", "few_shot_v4", "few_shot_v5", "few_shot_v6",
                             "few_shot_v7", "zero_shot_identify"])
    c.add_argument("--output", required=True, help="output parquet path")

    e = sub.add_parser("evaluate", help="Compute metrics and report")
    e.add_argument("--test-input", default="data/test-00000-of-00001_with_variants.parquet")
    e.add_argument("--zero-shot", help="path to zero-shot classified parquet")
    e.add_argument("--few-shot", help="path to few-shot classified parquet")
    e.add_argument("--few-shot-full", help="path to few_shot_full classified parquet")
    e.add_argument("--few-shot-v3", help="path to few_shot_v3 classified parquet")
    e.add_argument("--few-shot-v4", help="path to few_shot_v4 classified parquet")
    e.add_argument("--few-shot-v5", help="path to few_shot_v5 classified parquet")
    e.add_argument("--few-shot-v6", help="path to few_shot_v6 classified parquet")
    e.add_argument("--few-shot-v7", help="path to few_shot_v7 classified parquet")
    e.add_argument("--zero-shot-identify", help="path to open-task zero-shot parquet")
    e.add_argument("--output-dir", default="results")

    args = parser.parse_args()
    if args.command == "classify":
        cmd_classify(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
