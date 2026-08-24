"""Cross-row rotators for name / surname / relational-pair / strategy usage."""

from __future__ import annotations

from collections import Counter

DEFAULT_SURNAMES = [
    "Smith", "Davis", "Chen", "Fitzgerald", "Johnson", "Garcia",
    "Kim", "Patel", "Thompson", "Nguyen", "Brown", "Martinez",
]
DEFAULT_MALE_FIRST_NAMES = [
    "James", "David", "Michael", "Robert", "Thomas", "William",
    "Charles", "Daniel", "Matthew", "Andrew", "Joseph", "Kevin",
]
DEFAULT_FEMALE_FIRST_NAMES = [
    "Sarah", "Emma", "Jennifer", "Maria", "Jessica", "Laura",
    "Elizabeth", "Rebecca", "Anna", "Amanda", "Rachel", "Catherine",
]
DEFAULT_RELATIONAL_PAIRS = [
    ("brother", "sister"),
    ("uncle", "aunt"),
    ("father", "mother"),
    ("son", "daughter"),
    ("nephew", "niece"),
]


class _Rotator:
    def __init__(self, pool):
        self._pool = list(pool)
        self._counts = Counter({item: 0 for item in self._pool})

    def pick(self):
        chosen = min(self._pool, key=lambda x: (self._counts[x], self._pool.index(x)))
        self._counts[chosen] += 1
        return chosen

    def stats(self) -> dict:
        return dict(self._counts)


class SurnameRotator(_Rotator):
    def __init__(self, pool=None):
        super().__init__(pool or DEFAULT_SURNAMES)


class FirstNameRotator:
    def __init__(self, male_pool=None, female_pool=None):
        self._male = _Rotator(male_pool or DEFAULT_MALE_FIRST_NAMES)
        self._female = _Rotator(female_pool or DEFAULT_FEMALE_FIRST_NAMES)

    def pick_pair(self) -> tuple[str, str]:
        return self._male.pick(), self._female.pick()

    def stats(self) -> dict:
        return {"male": self._male.stats(), "female": self._female.stats()}


class RelationalPairRotator(_Rotator):
    def __init__(self, pool=None):
        super().__init__(pool or DEFAULT_RELATIONAL_PAIRS)


class StrategyCounter:
    def __init__(self):
        self._counts = Counter()

    def record(self, strategy: str):
        if strategy:
            self._counts[strategy] += 1

    def summary(self, total_so_far: int = None) -> str:
        total = total_so_far or sum(self._counts.values())
        if total == 0:
            return ""
        parts = []
        for strat, n in sorted(self._counts.items(), key=lambda x: -x[1]):
            parts.append(f"{strat}={int(100 * n / total)}%")
        return ", ".join(parts)

    def stats(self) -> dict:
        return dict(self._counts)
