import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rotators import (
    SurnameRotator, FirstNameRotator, RelationalPairRotator, StrategyCounter,
    DEFAULT_SURNAMES, DEFAULT_RELATIONAL_PAIRS,
)


def test_surname_rotator_uniform():
    r = SurnameRotator()
    picks = [r.pick() for _ in range(len(DEFAULT_SURNAMES))]
    assert set(picks) == set(DEFAULT_SURNAMES)


def test_surname_rotator_balanced_over_many():
    r = SurnameRotator()
    counts = {}
    for _ in range(60):
        s = r.pick()
        counts[s] = counts.get(s, 0) + 1
    values = list(counts.values())
    assert max(values) - min(values) <= 1


def test_first_name_rotator_separate_pools():
    r = FirstNameRotator()
    pair1 = r.pick_pair()
    pair2 = r.pick_pair()
    assert pair1[0] != pair2[0]
    assert pair1[1] != pair2[1]
    stats = r.stats()
    assert "male" in stats and "female" in stats


def test_relational_rotator_cycles():
    r = RelationalPairRotator()
    picks = [r.pick() for _ in range(len(DEFAULT_RELATIONAL_PAIRS))]
    assert set(picks) == set(DEFAULT_RELATIONAL_PAIRS)


def test_strategy_counter_summary():
    c = StrategyCounter()
    c.record("pronoun_swap")
    c.record("pronoun_swap")
    c.record("adjective")
    s = c.summary()
    assert "pronoun_swap" in s
    assert "adjective" in s


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
