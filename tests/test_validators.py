import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from validators import (
    word_edit_distance, check_edit_distance, check_formatting_preserved,
    count_sentence_boundaries, check_no_extra_sentence, check_variants_differ,
    check_variants_edited, check_referent_preserved, check_no_pos_change_symbolic,
    validate_row,
)


def test_word_edit_distance():
    assert word_edit_distance("the cat sat", "the cat sat") == 0
    assert word_edit_distance("the cat sat", "the dog sat") == 1
    assert word_edit_distance("the cat sat", "the cat") == 1
    assert word_edit_distance("", "hello world") == 2


def test_check_edit_distance_within_limits():
    src = "The doctor said they would arrive soon."
    var = "The doctor said he would arrive soon."
    ok, msg = check_edit_distance(src, var)
    assert ok is True
    assert msg == ""


def test_check_edit_distance_hard_fail():
    src = "The doctor said hi."
    var = "The doctor said hi and then added that he was going to the store to buy groceries for dinner tonight."
    ok, msg = check_edit_distance(src, var)
    assert ok is False
    assert "hard limit" in msg


def test_check_formatting_preserved_weird_spacing():
    src = "I' m tired."
    variant_bad = "I'm tired and he is too."
    ok, msg = check_formatting_preserved(src, variant_bad)
    assert ok is False
    assert "formatting drift" in msg


def test_check_formatting_preserved_good():
    src = "I' m tired."
    variant_ok = "He' s tired."
    ok, _ = check_formatting_preserved(src, variant_ok)
    assert ok is True


def test_count_sentence_boundaries():
    assert count_sentence_boundaries("hello world") == 0
    assert count_sentence_boundaries("Hello. World is nice.") == 1
    assert count_sentence_boundaries("Hi! How are you? I am fine.") == 2


def test_check_no_extra_sentence():
    src = "The doctor said hi."
    variant_ok = "The doctor, Mr. Davis, said hi."
    ok, _ = check_no_extra_sentence(src, variant_ok)
    assert ok is True

    variant_bad = "The doctor said hi. He was kind."
    ok, msg = check_no_extra_sentence(src, variant_bad)
    assert ok is False
    assert "sentence boundaries" in msg


def test_check_variants_differ():
    ok, _ = check_variants_differ("he said hi", "she said hi")
    assert ok is True
    ok, msg = check_variants_differ("he said hi", "he said hi")
    assert ok is False
    assert "identical" in msg


def test_check_variants_edited():
    ok, _ = check_variants_edited("orig", "masc", "fem")
    assert ok is True
    ok, msg = check_variants_edited("orig", "orig", "orig")
    assert ok is False


def test_check_referent_preserved_noun_strategy():
    src = "The doctor said hello."
    var_ok = "The doctor, Mr. Davis, said hello."
    ok, _ = check_referent_preserved(src, var_ok, "doctor", "title_insertion")
    assert ok is True


def test_check_referent_preserved_deletion():
    src = "The doctor said hello."
    var_bad = "The man, Mr. Davis, said hello."
    ok, msg = check_referent_preserved(src, var_bad, "doctor", "title_insertion")
    assert ok is False


def test_check_referent_preserved_swap_ok():
    src = "The waiter arrived."
    var = "The waitress arrived."
    ok, _ = check_referent_preserved(src, var, "waiter", "referent_swap")
    assert ok is True


def test_check_no_pos_change_noun_variant():
    src = "The local came to meet us."
    var_bad = "The local man came to meet us."
    ok, msg = check_no_pos_change_symbolic(src, var_bad, "local")
    assert ok is False
    assert "POS change" in msg


def test_check_no_pos_change_ok():
    src = "The local man came to meet us."
    var = "The local man came to meet us, sir."
    ok, _ = check_no_pos_change_symbolic(src, var, "local")
    assert ok is True


def test_validate_row_all_pass():
    row = {
        "EN_source_sentence": "The teacher said they would grade the exams.",
        "EN_masculine_sentence": "The teacher said he would grade the exams.",
        "EN_feminine_sentence": "The teacher said she would grade the exams.",
        "referent": "teacher",
        "disambiguation_strategy": "pronoun_swap",
    }
    result = validate_row(row)
    assert result["passed"] is True
    assert result["messages"] == []


def test_validate_row_identical_fails():
    row = {
        "EN_source_sentence": "hello",
        "EN_masculine_sentence": "hi there",
        "EN_feminine_sentence": "hi there",
        "referent": "x",
        "disambiguation_strategy": "pronoun_swap",
    }
    result = validate_row(row)
    assert result["passed"] is False
    assert any("identical" in m for m in result["messages"])


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
