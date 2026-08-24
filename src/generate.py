import argparse
import json
import os
import time

import anthropic
import pandas as pd


RESOURCES_DIR = "data/resources"


def load_reference_material():
    """Load gender lexicon and term lists for inclusion in the prompt."""
    with open(f"{RESOURCES_DIR}/lexical_gender_nouns.md", "r") as f:
        lexicon = f.read()

    lists = {}
    for fname in [
        "LLM_female_list.txt",
        "LLM_male_list.txt",
        "female_embedding_list.txt",
        "male_embedding_list.txt",
    ]:
        with open(f"{RESOURCES_DIR}/{fname}", "r") as f:
            lists[fname] = f.read().strip()

    return lexicon, lists


SURNAMES = [
    "Smith", "Davis", "Chen", "Fitzgerald", "Johnson", "Garcia",
    "Kim", "Patel", "Thompson", "Nguyen", "Brown", "Martinez",
]

FEW_SHOT_EXAMPLES = [
    {
        "referent": "teacher",
        "original": "The teacher said they would grade the exams over the weekend.",
        "masculine": "The teacher said he would grade the exams over the weekend.",
        "feminine": "The teacher said she would grade the exams over the weekend.",
        "strategy": "pronoun_swap",
    },
    {
        "referent": "teen",
        "original": "Your teen will feel better about themselves as a result.",
        "masculine": "Your teen will feel better about himself as a result.",
        "feminine": "Your teen will feel better about herself as a result.",
        "strategy": "pronoun_swap",
    },
    {
        "referent": "performer",
        "original": "The performer captivated the audience with an incredible solo.",
        "masculine": "The performer captivated the audience with his incredible solo.",
        "feminine": "The performer captivated the audience with her incredible solo.",
        "strategy": "pronoun_insertion",
    },
    {
        "referent": "photographer",
        "original": "Working with a renowned photographer and a specialist aerial film crew, we highlighted the incredible range of services.",
        "masculine": "Working with a renowned photographer and his specialist aerial film crew, we highlighted the incredible range of services.",
        "feminine": "Working with a renowned photographer and her specialist aerial film crew, we highlighted the incredible range of services.",
        "strategy": "pronoun_insertion",
    },
    {
        "referent": "opponent",
        "original": "I know you're a tough cookie, but you'll find I'm a worthy opponent.",
        "masculine": "I know you're a tough cookie, but you'll find he's a worthy opponent.",
        "feminine": "I know you're a tough cookie, but you'll find she's a worthy opponent.",
        "strategy": "pronoun_swap",
    },
    {
        "referent": "engineer",
        "original": "I'm a semi-professional producer, audio engineer, and session musician in the Seattle area.",
        "masculine": "He's a semi-professional producer, audio engineer, and session musician in the Seattle area.",
        "feminine": "She's a semi-professional producer, audio engineer, and session musician in the Seattle area.",
        "strategy": "pronoun_swap",
    },
    {
        "referent": "waiter",
        "original": "The waiter brought our drinks and asked if we were ready to order.",
        "masculine": "The waiter brought our drinks and asked if we were ready to order.",
        "feminine": "The waitress brought our drinks and asked if we were ready to order.",
        "strategy": "referent_swap",
        "masc_referent": "waiter",
        "fem_referent": "waitress",
    },
    {
        "referent": "salesperson",
        "original": "The salesperson said they would follow up with the client next week.",
        "masculine": "The salesman said he would follow up with the client next week.",
        "feminine": "The saleswoman said she would follow up with the client next week.",
        "strategy": "combination",
        "masc_referent": "salesman",
        "fem_referent": "saleswoman",
    },
    {
        "referent": "actor",
        "original": "The lead actor delivered a stunning performance at last night's premiere.",
        "masculine": "The lead actor delivered a stunning performance at last night's premiere.",
        "feminine": "The lead actress delivered a stunning performance at last night's premiere.",
        "strategy": "referent_swap",
        "masc_referent": "actor",
        "fem_referent": "actress",
    },
    {
        "referent": "host",
        "original": "Their host for the evening greeted them warmly at the door.",
        "masculine": "Their host for the evening greeted them warmly at the door.",
        "feminine": "Their hostess for the evening greeted them warmly at the door.",
        "strategy": "referent_swap",
        "masc_referent": "host",
        "fem_referent": "hostess",
    },
    {
        "referent": "firefighter",
        "original": "A firefighter rescued the family from the burning building.",
        "masculine": "A fireman rescued the family from the burning building.",
        "feminine": "A firewoman rescued the family from the burning building.",
        "strategy": "referent_swap",
        "masc_referent": "fireman",
        "fem_referent": "firewoman",
    },
    {
        "referent": "heir",
        "original": "The wealthy family named their eldest child as their heir.",
        "masculine": "The wealthy family named their eldest child as their heir.",
        "feminine": "The wealthy family named their eldest child as their heiress.",
        "strategy": "referent_swap",
        "masc_referent": "heir",
        "fem_referent": "heiress",
    },
    {
        "referent": "enemy",
        "original": "The person who really understands me is my most feared enemy.",
        "masculine": "The man who really understands me is my most feared enemy.",
        "feminine": "The woman who really understands me is my most feared enemy.",
        "strategy": "context_modifier",
    },
    {
        "referent": "captain",
        "original": "That person, the captain of the team, was very kind to the new players.",
        "masculine": "That man, the captain of the team, was very kind to the new players.",
        "feminine": "That woman, the captain of the team, was very kind to the new players.",
        "strategy": "context_modifier",
    },
    {
        "referent": "dancer",
        "original": "The elderly person who was a professional dancer demonstrated a few steps.",
        "masculine": "The elderly gentleman who was a professional dancer demonstrated a few steps.",
        "feminine": "The elderly lady who was a professional dancer demonstrated a few steps.",
        "strategy": "context_modifier",
    },
    {
        "referent": "investigator",
        "original": "The investigator was trained in recording the fingerprints before the start of the study.",
        "masculine": "The investigator, Mr. Fitzgerald, was trained in recording the fingerprints before the start of the study.",
        "feminine": "The investigator, Mrs. Fitzgerald, was trained in recording the fingerprints before the start of the study.",
        "strategy": "title_insertion",
    },
    {
        "referent": "chef",
        "original": "The recent story about a pampered celebrity chef left me very annoyed.",
        "masculine": "The recent story about a pampered celebrity chef, Mr. Davis, left me very annoyed.",
        "feminine": "The recent story about a pampered celebrity chef, Mrs. Davis, left me very annoyed.",
        "strategy": "title_insertion",
    },
    {
        "referent": "officer",
        "original": "They packed us like sardines with a NKVD officer inside and soldiers outside the cars.",
        "masculine": "They packed us like sardines with a male NKVD officer inside and soldiers outside the cars.",
        "feminine": "They packed us like sardines with a female NKVD officer inside and soldiers outside the cars.",
        "strategy": "adjective",
    },
    {
        "referent": "librarian",
        "original": "If you are a librarian, you can embed the wiki content into your site.",
        "masculine": "If you are a male librarian, you can embed the wiki content into your site.",
        "feminine": "If you are a female librarian, you can embed the wiki content into your site.",
        "strategy": "adjective",
    },
    {
        "referent": "chef",
        "original": "The recent story about a pampered celebrity chef left me very annoyed.",
        "masculine": "The recent story about a pampered celebrity chef - a man - left me very annoyed.",
        "feminine": "The recent story about a pampered celebrity chef - a woman - left me very annoyed.",
        "strategy": "appositive_dash",
    },
    {
        "referent": "bystander",
        "original": "As I start speaking about it I cry, one bystander told Health-e News.",
        "masculine": "As I start speaking about it I cry, one bystander - a man - told Health-e News.",
        "feminine": "As I start speaking about it I cry, one bystander - a woman - told Health-e News.",
        "strategy": "appositive_dash",
    },
    {
        "referent": "janitor",
        "original": "He does his research and pays off a janitor at the hospital.",
        "masculine": "He does his research and pays off a janitor, his brother, at the hospital.",
        "feminine": "He does his research and pays off a janitor, his sister, at the hospital.",
        "strategy": "relational_insertion",
    },
    {
        "referent": "neighbor",
        "original": "The neighbor came by to borrow some sugar this morning.",
        "masculine": "The neighbor, my uncle, came by to borrow some sugar this morning.",
        "feminine": "The neighbor, my aunt, came by to borrow some sugar this morning.",
        "strategy": "relational_insertion",
    },
    {
        "referent": "publisher",
        "original": "A quote from this publisher's leading author flows through the piece, providing a sense of rhythm.",
        "masculine": "A quote from this publisher's leading author flows through the piece, providing him, the publisher, a sense of rhythm.",
        "feminine": "A quote from this publisher's leading author flows through the piece, providing her, the publisher, a sense of rhythm.",
        "strategy": "pronoun_insertion",
    },
    {
        "referent": "enthusiast",
        "original": "Whether you are a seasoned professional or a budding enthusiast, you will not be disappointed!",
        "masculine": "Whether you are a seasoned professional or a budding enthusiast, sir, you will not be disappointed!",
        "feminine": "Whether you are a seasoned professional or a budding enthusiast, ma'am, you will not be disappointed!",
        "strategy": "sir_maam",
    },
    {
        "referent": "coach",
        "original": "Yes, coach, I understand.",
        "masculine": "Yes, coach, sir, I understand.",
        "feminine": "Yes, coach, ma'am, I understand.",
        "strategy": "sir_maam",
    },
    {
        "referent": "counselor",
        "original": "We have 4 Licensed Mental Health Counselors on staff.",
        "masculine": "We have 4 Licensed Mental Health Counselors on staff, led by Mr. Thompson.",
        "feminine": "We have 4 Licensed Mental Health Counselors on staff, led by Mrs. Thompson.",
        "strategy": "title_insertion",
    },
]


def build_system_prompt(lexicon, lists):
    """Build the system prompt with reference material and few-shot examples."""
    few_shot_text = ""
    for ex in FEW_SHOT_EXAMPLES:
        extra = ""
        if "masc_referent" in ex:
            extra = f'\n  Masc_referent: {ex["masc_referent"]}\n  Fem_referent: {ex["fem_referent"]}'
        few_shot_text += f"""
Example:
  Referent: {ex["referent"]}
  Original: {ex["original"]}
  Masculine: {ex["masculine"]}
  Feminine: {ex["feminine"]}
  Strategy: {ex["strategy"]}{extra}
"""

    surname_list = ", ".join(SURNAMES)
    return f"""You are a linguist generating gender-disambiguated contrastive sentence pairs for a research dataset.

TASK: Given an English sentence and a referent word, produce two variants where the REFERENT's gender is unambiguously masculine or feminine.

PRINCIPLE: Pick the strategy that reads most naturally for THIS specific sentence. Naturalness matters more than perfect meaning preservation - small semantic shifts and voice changes are fine if the result reads better. Do not default to any single strategy.

HARD RULES:
1. The gender of the REFERENT must be clear. Do not disambiguate a different word.
2. NEVER add a second sentence (after a period). You MAY add short clauses within the sentence using commas or dashes (e.g., ", he said" or "- she was clear about it").
3. Both variants MUST differ from the original AND from each other.
4. Preserve original formatting and punctuation style exactly (e.g., "I' m" stays "I' m").
5. Make sure pronouns you change actually refer to the referent, not to someone else in the sentence.
6. SYMMETRIC 1st-person swap: if the referent is the speaker ("I'm a doctor") and you swap "I" to "he"/"she", also swap any OTHER "I"/"me"/"my" in the sentence that refers to the same speaker. Example: "I know you're tough, but I'm a worthy opponent" -> "He knows you're tough, but he's a worthy opponent".
7. PREFER EMBEDDED edits over tacked-on dash clauses. Before adding "- he needs X" or "- she was Y", first try to integrate the pronoun into the existing sentence structure (possessive, subject swap, pronoun insertion). Dash clauses are a last resort when no natural in-sentence edit works - and if you do use them, prefer pure appositives like "- a man -" over action clauses like "- he sounded urgent".
8. If the referent appears as "you" (the addressee), disambiguate via another signal (sir/ma'am if appropriate, a nearby title, a male/female adjective) - do NOT change "you" to "he/she" since that changes the addressee.

STRATEGIES:

DECISION RULE: First check the gender lexicon. If the referent has a natural gendered pair (waiter/waitress, actor/actress, host/hostess, heir/heiress, salesperson/salesman/saleswoman, firefighter/fireman/firewoman, etc.), use REFERENT SWAP (strategy 1). Otherwise, use PRONOUN STRATEGY (strategy 2). Only consider strategies 3-8 when neither fits naturally.

FORBIDDEN REFERENT SWAPS: Do NOT use referent_swap for these pairs - their gendered forms are connotatively asymmetric and the masculine form is often unmarked or identical to the source:
  - hero / heroine
  - master / mistress
  - confidant / confidante
If the referent is one of these (or its gendered counterpart), pick a different strategy. If no other strategy fits, you may keep the masculine variant identical to the source and only edit the feminine variant via a different strategy - but never use referent_swap on these specific pairs.

1. REFERENT SWAP (HIGHEST PRIORITY when a natural, non-forbidden gendered pair exists): Replace the referent with its natural gendered equivalent from the reference lexicon. This is the cleanest disambiguation. Examples: waiter -> waiter/waitress, salesperson -> salesman/saleswoman, actor -> actor/actress, firefighter -> fireman/firewoman, heir -> heir/heiress, host -> host/hostess, chairperson -> chairman/chairwoman, businessperson -> businessman/businesswoman, police officer -> policeman/policewoman. ALWAYS scan the reference lexicon and term lists before picking another strategy. When you use this, set masc_referent and fem_referent in the output.

2. PRONOUN swap/insertion (PREFERRED when no natural referent pair exists): Swap existing pronouns (they -> he/she, themselves -> himself/herself) or insert a gendered pronoun near the referent (e.g., "the boss's plan" -> "his plan"/"her plan", "a photographer and a crew" -> "a photographer and his crew"/"a photographer and her crew"). You may also use a comma-set appositive pronoun like "providing him, the publisher, a sense of rhythm" / "providing her, the publisher, a sense of rhythm". Prefer commas over dashes for this pattern.

3. CONTEXT MODIFIER: Swap a nearby NEUTRAL noun with a gendered one, while keeping the referent noun intact. Works when the sentence contains an upstream word like "person", "individual", "someone", "one" that refers to the same entity as the referent. Example: "That person, the captain, was kind" -> "That man/woman, the captain, was kind" (referent "captain" stays, "person" becomes "man"/"woman"). Also works with descriptor swaps like "the elderly person who was a dancer" -> "the elderly gentleman/lady who was a dancer". CAUTION: do NOT turn the referent itself into an adjective by attaching "man/woman" after it. "the local" -> "the local man" is BAD because "local" becomes an adjective.

SECONDARY (for variety - use roughly EQUAL proportions across these 5):
4. RELATIONAL INSERTION: Insert a gendered family relationship AS AN APPOSITIVE that REFERS TO THE REFERENT ITSELF. The relation MUST equate to the referent - it is the same person. Example: "pays off a janitor" -> "pays off a janitor, his brother," (the brother IS the janitor). WRONG example: "become a mechanic, like your uncle" - here the uncle is a different person. Good pairs: brother/sister, uncle/aunt, father/mother, son/daughter, nephew/niece.

5. TITLE INSERTION: Add "Mr./Mrs." + an invented surname as an appositive near the referent. Example: "The investigator, Mr. Fitzgerald, was trained..." / "The investigator, Mrs. Fitzgerald, was trained...". Available surnames: {surname_list}.

6. SIR/MA'AM: When the referent is the addressee ("you're a good coach" or "Yes, enthusiast"), add sir/ma'am as a vocative. Example: "Whether you are a budding enthusiast" -> "Whether you are a budding enthusiast, sir," / "...ma'am,".

7. ADJECTIVE: Add "male"/"female" before the referent. Example: "a male officer"/"a female officer".

8. APPOSITIVE DASH: Insert "- a man -"/"- a woman -" as a dashed appositive. Example: "the chef - a man - left" / "the chef - a woman - left".

DECISION ORDER:
1. Does the referent have a natural gendered pair in the lexicon? → USE REFERENT SWAP.
2. Does the sentence contain a pronoun that refers to the referent (they/their/themselves), or a nearby possessive slot? → USE PRONOUN SWAP/INSERTION.
3. Otherwise, pick from these 6 secondary strategies roughly equally:
   - CONTEXT MODIFIER (swap "person"/"one"/"individual" with "man"/"woman") - when upstream neutral noun exists
   - RELATIONAL INSERTION (his brother / her sister) - natural for known people
   - TITLE INSERTION (Mr./Mrs. Surname) - good for formal/professional contexts
   - SIR/MA'AM - only when referent is addressee
   - ADJECTIVE (male / female) - use moderately
   - APPOSITIVE DASH (- a man -) - use moderately

TARGET: across outputs, aim for roughly 8-15% usage of each secondary strategy. Adjective and appositive_dash tend to get overused - before defaulting to either, check whether CONTEXT MODIFIER, RELATIONAL, TITLE, or SIR/MA'AM would fit naturally.

Do NOT refuse a sentence just to balance - naturalness always wins. If no strategy fits naturally, use whichever is least awkward.

9. COMBINATION: Combine two or more strategies when needed.

REFERENCE - Gendered noun pairs:
{lexicon}

REFERENCE - Gendered term lists:
Female-associated: {lists["LLM_female_list.txt"]}
Male-associated: {lists["LLM_male_list.txt"]}
Female embeddings: {lists["female_embedding_list.txt"]}
Male embeddings: {lists["male_embedding_list.txt"]}

EXAMPLES:
{few_shot_text}

OUTPUT FORMAT: Respond with ONLY a JSON object (no markdown):
{{"masculine_sentence": "...", "feminine_sentence": "...", "strategy": "pronoun_swap|pronoun_insertion|referent_swap|context_modifier|relational_insertion|title_insertion|sir_maam|adjective|appositive_dash|combination", "masc_referent": "...", "fem_referent": "..."}}

The masc_referent and fem_referent fields should contain the referent word as it appears in the masculine/feminine sentence. In most cases these will be the same as the original referent. Only when using referent_swap will they differ.
"""


def generate_contrastive_pair(client, model, system_prompt, referent, sentence, max_retries=3):
    """Call Claude API to generate masculine/feminine variants with retry on rate limits."""
    user_msg = f"Referent: {referent}\nOriginal sentence: {sentence}"

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )

            text = response.content[0].text.strip()
            # Handle potential markdown code block wrapping
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0].strip()

            # Extract first JSON object if model returns extra text
            start = text.index("{")
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[start:i+1]
                        break

            return json.loads(text)
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 30  # 30s, 60s, 120s
            print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)

    raise RuntimeError(f"Rate limited after {max_retries} retries")


def main():
    parser = argparse.ArgumentParser(
        description="Generate contrastive gender sentence pairs"
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="train",
        help="Which dataset split to process",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Claude model to use",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of rows to sample (for prototyping)",
    )
    parser.add_argument(
        "--output-dir",
        default="generated",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    splits = (
        ["train", "validation", "test"] if args.split == "all" else [args.split]
    )

    client = anthropic.Anthropic()
    lexicon, lists = load_reference_material()
    system_prompt = build_system_prompt(lexicon, lists)

    os.makedirs(args.output_dir, exist_ok=True)

    for split in splits:
        print(f"\n=== Processing {split} split ===")
        df = pd.read_parquet(f"data/{split}-00000-of-00001.parquet")

        if args.sample is not None:
            df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
            print(f"Sampled {len(df)} rows")

        masculine_sentences = []
        feminine_sentences = []
        strategies = []
        errors = []

        for idx, row in df.iterrows():
            referent = row["referent"]
            sentence = row["EN_source_sentence"]
            print(f"  [{len(masculine_sentences)+1}/{len(df)}] {referent}: {sentence[:60]}...")

            try:
                result = generate_contrastive_pair(
                    client, args.model, system_prompt, referent, sentence
                )
                masculine_sentences.append(result["masculine_sentence"])
                feminine_sentences.append(result["feminine_sentence"])
                strategies.append(result.get("strategy", "unknown"))
            except Exception as e:
                print(f"    ERROR: {e}")
                masculine_sentences.append(None)
                feminine_sentences.append(None)
                strategies.append(None)
                errors.append({"index": idx, "referent": referent, "error": str(e)})

        df = df.copy()
        df["EN_masculine_sentence"] = masculine_sentences
        df["EN_feminine_sentence"] = feminine_sentences
        df["disambiguation_strategy"] = strategies

        sample_suffix = f"_sample{args.sample}" if args.sample is not None else ""
        model_suffix = args.model.replace("claude-", "").split("-202")[0]
        out_path = os.path.join(
            args.output_dir, f"{split}{sample_suffix}_{model_suffix}.parquet"
        )
        df.to_parquet(out_path, index=False)
        print(f"  Saved to {out_path}")

        if errors:
            err_path = os.path.join(
                args.output_dir,
                f"{split}{sample_suffix}_{model_suffix}_errors.json",
            )
            with open(err_path, "w") as f:
                json.dump(errors, f, indent=2)
            print(f"  {len(errors)} errors saved to {err_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
