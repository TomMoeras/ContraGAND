import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from grounding import (
    parse_lexicon, detect_referent_pair, detect_addressee,
    detect_first_person_referent, detect_existing_neutral_pronoun, analyze,
    FORBIDDEN_SWAP_PAIRS,
)


LEX_SAMPLE = """
| Masculine | Feminine | Neutral |
|-----------|----------|---------|
| waiter | waitress | server |
| salesman | saleswoman | salesperson |
| fireman | firewoman | firefighter |
| wizard | witch | - |
| actor | actress | - |
"""


LEX_FORBIDDEN_SAMPLE = """
| Masculine | Feminine | Neutral |
|-----------|----------|---------|
| waiter | waitress | server |
| hero | heroine | - |
| master | mistress | - |
| confidant | confidante | - |
"""


def test_parse_lexicon():
    lookup = parse_lexicon(LEX_SAMPLE)
    assert lookup["waiter"] == ("waiter", "waitress")
    assert lookup["waitress"] == ("waiter", "waitress")
    assert lookup["server"] == ("waiter", "waitress")
    assert lookup["salesperson"] == ("salesman", "saleswoman")
    assert lookup["firefighter"] == ("fireman", "firewoman")
    assert lookup["wizard"] == ("wizard", "witch")


def test_detect_referent_pair():
    lookup = parse_lexicon(LEX_SAMPLE)
    assert detect_referent_pair("waiter", lookup) == ("waiter", "waitress")
    assert detect_referent_pair("Waiter", lookup) == ("waiter", "waitress")
    assert detect_referent_pair("doctor", lookup) is None


def test_detect_addressee_vocative_start():
    assert detect_addressee("Yes, coach, I understand.", "coach") is True
    assert detect_addressee("Hey, buddy, how are you?", "buddy") is True
    assert detect_addressee("Look, officer, I swear.", "officer") is True


def test_detect_addressee_trailing_vocative():
    assert detect_addressee("I'm fine, coach.", "coach") is True
    assert detect_addressee("Step back, officer!", "officer") is True
    assert detect_addressee("Thank you, doctor.", "doctor") is True


def test_detect_addressee_third_person():
    assert detect_addressee("The coach said he would start practice.", "coach") is False
    assert detect_addressee("My doctor is very thorough.", "doctor") is False


def test_detect_first_person_referent():
    assert detect_first_person_referent("I'm a doctor now.", "doctor") is True
    assert detect_first_person_referent("I am a teacher at the high school.", "teacher") is True
    assert detect_first_person_referent("As a journalist, I covered the story.", "journalist") is True


def test_detect_first_person_referent_negative():
    assert detect_first_person_referent("The doctor said hello.", "doctor") is False
    assert detect_first_person_referent("She is a teacher.", "teacher") is False


def test_detect_existing_neutral_pronoun():
    s = "The teacher said they would grade the exams."
    assert detect_existing_neutral_pronoun(s, "teacher") == {"they"}

    s = "Your teen will feel better about themselves as a result."
    assert detect_existing_neutral_pronoun(s, "teen") == {"themselves"}

    s = "The doctor did not call back."
    assert detect_existing_neutral_pronoun(s, "doctor") == set()


def test_analyze_prioritises_referent_swap():
    lookup = parse_lexicon(LEX_SAMPLE)
    row = {"referent": "waiter", "EN_source_sentence": "Yes, waiter, I would like the check."}
    hint = analyze(row, lookup)
    assert hint.recommended_strategy == "referent_swap"
    assert hint.masc_referent == "waiter"
    assert hint.fem_referent == "waitress"


def test_analyze_routes_addressee_when_no_pair():
    lookup = parse_lexicon(LEX_SAMPLE)
    row = {"referent": "coach", "EN_source_sentence": "Yes, coach, I understand the play."}
    hint = analyze(row, lookup)
    assert hint.recommended_strategy == "sir_maam"


def test_analyze_routes_pronoun_swap_when_neutral_pronoun():
    lookup = parse_lexicon(LEX_SAMPLE)
    row = {"referent": "teacher", "EN_source_sentence": "The teacher said they would grade the exams."}
    hint = analyze(row, lookup)
    assert hint.recommended_strategy == "pronoun_swap"
    assert "they" in hint.note


def test_analyze_routes_first_person():
    lookup = parse_lexicon(LEX_SAMPLE)
    row = {"referent": "engineer", "EN_source_sentence": "I'm a semi-professional audio engineer in Seattle."}
    hint = analyze(row, lookup)
    assert hint.recommended_strategy == "pronoun_swap"
    assert hint.signals.get("first_person") is True


def test_analyze_empty_hint():
    lookup = parse_lexicon(LEX_SAMPLE)
    row = {"referent": "buddy", "EN_source_sentence": "My buddy came over yesterday."}
    hint = analyze(row, lookup)
    assert hint.recommended_strategy is None
    assert hint.render() == ""


def test_hint_render():
    from grounding import Hint
    h = Hint(recommended_strategy="referent_swap", masc_referent="waiter", fem_referent="waitress")
    rendered = h.render()
    assert "RECOMMENDED_STRATEGY: referent_swap" in rendered
    assert "MASC_REFERENT: waiter" in rendered
    assert "FEM_REFERENT: waitress" in rendered


def test_forbidden_pairs_constant_includes_expected():
    assert ("hero", "heroine") in FORBIDDEN_SWAP_PAIRS
    assert ("master", "mistress") in FORBIDDEN_SWAP_PAIRS
    assert ("confidant", "confidante") in FORBIDDEN_SWAP_PAIRS


def test_analyze_skips_forbidden_pair_hero():
    lookup = parse_lexicon(LEX_FORBIDDEN_SAMPLE)
    row = {"referent": "hero", "EN_source_sentence": "The hero saved the village."}
    hint = analyze(row, lookup)
    assert hint.recommended_strategy != "referent_swap"
    assert hint.signals.get("forbidden_pair") is True


def test_analyze_skips_forbidden_pair_master():
    lookup = parse_lexicon(LEX_FORBIDDEN_SAMPLE)
    row = {"referent": "master", "EN_source_sentence": "I am the master of my own destiny."}
    hint = analyze(row, lookup)
    assert hint.recommended_strategy != "referent_swap"


def test_analyze_skips_forbidden_pair_confidant():
    lookup = parse_lexicon(LEX_FORBIDDEN_SAMPLE)
    row = {"referent": "confidant", "EN_source_sentence": "She trusted her confidant deeply."}
    hint = analyze(row, lookup)
    assert hint.recommended_strategy != "referent_swap"


def test_analyze_non_forbidden_pair_still_swaps():
    lookup = parse_lexicon(LEX_FORBIDDEN_SAMPLE)
    row = {"referent": "waiter", "EN_source_sentence": "The waiter brought our drinks."}
    hint = analyze(row, lookup)
    assert hint.recommended_strategy == "referent_swap"
    assert hint.masc_referent == "waiter"
    assert hint.fem_referent == "waitress"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
