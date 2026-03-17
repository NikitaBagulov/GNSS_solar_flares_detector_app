from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Literal

ExistingDataPolicy = Literal["skip", "overwrite", "validate"]
ModuleName = Literal["download", "preprocess", "index", "plot"]


@dataclass(frozen=True)
class RunConfig:
    existing_data_policy: ExistingDataPolicy = "validate"
    skip_modules: FrozenSet[ModuleName] = frozenset()
    overwrite_modules: FrozenSet[ModuleName] = frozenset()
    validate_modules: FrozenSet[ModuleName] = frozenset()

    def policy_for(self, module: ModuleName) -> ExistingDataPolicy:
        if module in self.skip_modules:
            return "skip"
        if module in self.overwrite_modules:
            return "overwrite"
        if module in self.validate_modules:
            return "validate"
        return self.existing_data_policy


def as_module_set(values: Iterable[str]) -> FrozenSet[ModuleName]:
    return frozenset(values)  # type: ignore[return-value]

