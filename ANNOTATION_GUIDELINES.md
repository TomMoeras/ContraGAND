# ContraGAND test-set annotation guidelines (human inter-annotator agreement)

## What this is

We built a corpus (ContraGAND) of gender-ambiguous English sentences. For each source sentence and a fixed referent word (an occupation/role/relational noun such as *doctor*, *author*, *colleague*), an automatic pipeline produced two rewritten variants:

- a **masculine variant**, where the referent is now unambiguously male, and
- a **feminine variant**, where the referent is now unambiguously female,

changing as little else as possible. Every variant is also tagged with the strategy used to make the edit (e.g. swapping a pronoun, adding a title).

Your job is to independently re-rate the items. You will make one **ACCEPT / REJECT** decision per item.

Please annotate independently. Do not discuss items with the other annotator(s) while you work, and do not look at the judge's scores or the previous review.

## The unit you are judging

Each row you rate is one pair for one referent. You will see four things:

- **Referent**: the word whose gender the edit must make clear.
- **Source sentence**: the original, deliberately gender-ambiguous sentence.
- **Masculine variant**: the rewrite that should make the referent clearly male.
- **Feminine variant**: the rewrite that should make the referent clearly female.

You judge the pair as a whole: both variants, together, against the source, for that one referent. A single ACCEPT/REJECT applies to the pair.

## The criteria

Read the pair with these five questions in mind.

1. **Referent disambiguated.** Do the edits make the referent itself clearly gendered, rather than gendering some other person in the sentence? The masculine variant should read unambiguously male for the referent, and the feminine variant unambiguously female for the referent.
   - If the referent is the person being addressed ("you" / a direct address), adding *sir* / *ma'am* or a title does count as disambiguating the referent.
   - If a pronoun is inserted with a short appositive that ties it to the referent (e.g. "providing him, the publisher, a sense of rhythm" for the referent *publisher*), that does count as disambiguating.
2. **Correctness.** Are both variants grammatically correct English?
3. **Naturalness.** Do both variants read naturally, not awkward, forced, or stilted?
4. **No extra sentence.** The edit must not add a whole new sentence after a period. Short additions set off by commas or dashes are fine (e.g. "the author, Mr. Chen, follows..."). A new standalone sentence (e.g. "... the shelter. He worked hard.") is not.
5. **Consistency.** Are the masculine and feminine edits symmetric / balanced — the same kind of change, in the same place, just flipped in gender? One side should not do something heavy while the other does something trivial.

## Your decision procedure

Work through this in order. As soon as a REJECT condition is met, you can stop and reject.

**Reject the pair if ANY of these is true:**

- **A. Unfixable source** — the source is broken, is multiple sentences mashed together, or the referent genuinely cannot be given a natural gender.
- **B. Part-of-speech change** — an edit turned the referent from a noun into a modifier (the "the local man" problem below).
- **C. Wrong thing gendered** — either variant fails to make the referent gendered; it genders someone/something else, or leaves the referent still ambiguous.
- **D. Extra full sentence** — a new standalone sentence was added after a period (short comma/dash clauses are fine).

If none of A–D fires, weigh criteria 2, 3, and 5 (correctness, naturalness, consistency):

- **ACCEPT** if the pair is clearly correct, reads naturally, and the two edits are symmetric, with at most minor blemishes. This is the "I would be comfortable using this as gold-standard data" bar.
- **REJECT** if the pair has clear grammatical errors, reads awkwardly or forced, or the masculine and feminine edits are noticeably asymmetric or unbalanced.

When you are genuinely on the fence after applying the rule, lean toward the decision that matches "would I trust this as clean labelled data?" and record a short note explaining the hesitation.

## Recognising valid edit strategies

Any of the following, done well, is a legitimate way to disambiguate the referent. Knowing them helps you tell a good edit from a wrong one. (You do not need to label the strategy; it is provided to you.)

| Strategy | What a good edit looks like |
|---|---|
| pronoun swap | an existing pronoun is flipped (*they* → *he* / *she*) |
| pronoun insertion | a gendered pronoun is added near the referent (*a photographer and his/her crew*) |
| context modifier | a nearby neutral word is gendered (*the person* → *the man* / *the woman*) |
| relational insertion | a gendered family relation is added as an appositive (*a janitor, his brother / her sister*) |
| title insertion | *Mr.* / *Mrs.* + a name is added (*the author, Mr. Chen, ...*) |
| sir / ma'am | *sir* / *ma'am* added as a form of address — only valid when the referent is the one being addressed |
| adjective | *male* / *female* placed before the referent (*a male / female colleague*) |
| appositive dash | *— a man —* / *— a woman —* inserted between dashes |
| referent swap | the referent word itself is replaced with its gendered form (*attendant* → *steward* / *stewardess*) |
| combination | two or more of the above |

A note on **referent swap**: pairs whose gendered forms carry lopsided connotations — *hero/heroine*, *master/mistress*, *confidant/confidante* — tend to be poor, because the masculine form is often the unmarked default while the feminine form reads as archaic or diminished. Treat those with extra scrutiny under the consistency criterion.

## Worked examples

### Accept

**Pronoun swap.** Referent *enthusiast*.
- Source: "I'm an IT Professional by day and an Electric Universe/Thunderbolts Project enthusiast the rest of the time."
- Masculine: "He's an IT Professional by day and ... enthusiast the rest of the time."
- Feminine: "She's an IT Professional by day and ... enthusiast the rest of the time."
- Verdict: **ACCEPT**. The referent (the speaker, who is the enthusiast) is clearly gendered; correct, natural, symmetric.

**Title insertion.** Referent *author*.
- Source: "The author follows 8 Milwaukee families through their daily struggle..."
- Masculine: "The author, Mr. Chen, follows 8 Milwaukee families..."
- Feminine: "The author, Mrs. Chen, follows 8 Milwaukee families..."
- Verdict: **ACCEPT**. Comma-set appositive (not a new sentence), symmetric, natural.

**Adjective.** Referent *colleague*.
- Source: "I haven't called a colleague today, but the one whom I love."
- Masculine: "I haven't called a male colleague today, but the one whom I love."
- Feminine: "I haven't called a female colleague today, but the one whom I love."
- Verdict: **ACCEPT**. Minimal, symmetric, grammatical.

**Referent swap.** Referent *attendant*.
- Source: "I was a flight attendant for 30 years..."
- Masculine: "I was a flight steward for 30 years..."
- Feminine: "I was a flight stewardess for 30 years..."
- Verdict: **ACCEPT**. Natural gendered pair, balanced.

### Reject

**Extra full sentence (rule D).** Referent *doctor*.
- Masculine: "The doctor examined the patient. He was very thorough."
- A brand-new standalone sentence was appended to carry the gender. Reject.

**Wrong thing gendered (rule C).** Referent *publisher*.
- Masculine: "The publisher's bankers, all men, promoted a layoff plan."
- The bankers were gendered, not the publisher. The referent is still ambiguous. Reject.

**Part-of-speech change (rule B).** Referent *local* (used as a noun, "the local").
- Masculine: "the local man" — *local* is now an adjective and *man* is the noun. Reject.

**Asymmetric / awkward vocative.** Referent *idiot* used inside an exclamation.
- Masculine: "...calculate the human mind, idiot, sir!"
- If the *sir* / *ma'am* is forced, or the referent is not actually the person being addressed, the edit is awkward and unbalanced. Reject (or flag as borderline in your note).

## How to record your answers

You will get a spreadsheet with one row per item, showing the referent, the source, and the two variants. Fill in two columns:

| Column | What to enter |
|---|---|
| `eval_human` | `accept` to accept the pair, `reject` to reject it. Nothing else. |
| `note` | (optional) a few words on why, especially for rejects and borderline calls, e.g. "extra sentence", "wrong entity gendered", "awkward sir". |

- Give exactly one decision for every row. Do not leave blanks.
- Rate every row in the set; do not skip.
- Work independently and do not change earlier answers based on later ones.

Return the completed sheet as you received it (same filename plus your initials, e.g. `..._AA.csv`).

## Quick reference (keep this visible while annotating)

**Reject if**: source unfixable | referent turned from noun to modifier | the referent (not someone else) is still ambiguous | a new full sentence was added. **Otherwise accept if**: grammatical + natural + the two edits are symmetric, with only minor blemishes.
