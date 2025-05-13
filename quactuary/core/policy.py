from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class PolicyTerms:
    per_occ_deductible: float = 0.0
    coinsurance:        float = 1.00          # 1 → 100% insurer share
    per_occ_limit:      Optional[float] = None
    agg_limit:          Optional[float] = None
    attachment:         float = 0.0         # for XoL layers
    coverage:           str = "OCC"       # OCC / CLAIMS‑MADE / etc.
    # TODO: policy dates, reinstatements, etc.
    # TODO: corridors
    # TODO: LoB
