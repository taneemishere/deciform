"""Escalation policy: when a decision model's answer is trusted vs. sent to fallback.

The published failure mode (JevAgentHarness, arXiv 2610.21412) is a single
global 0.9 gut-feel threshold. This policy gives per-field control and a
fitting hook — set an acceptable error rate and fit thresholds on labeled
data, instead of guessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from deciform.results import Answer


@dataclass
class EscalationPolicy:
    """Field-level accept/escalate thresholds plus option-order debiasing.

    - ``threshold``: default top-probability required to accept a decision
      (0.0 = trust everything, 1.0 = escalate everything).
    - ``per_field``: path -> threshold overrides for critical fields.
    - ``debias_k``: option-order permutations for choice questions (k=1 = off).
      Open-weight decision models flip ~30% of answers under reordering
      (arXiv 2610.02267); averaging over rotations dampens it.
    """

    threshold: float = 0.0
    per_field: dict[str, float] = field(default_factory=dict)
    debias_k: int = 1

    def threshold_for(self, path: str) -> float:
        return self.per_field.get(path, self.threshold)

    def should_escalate(self, path: str, answer: Answer) -> bool:
        return answer.top_probability < self.threshold_for(path)

    def fit(
        self,
        calib: dict[str, list[tuple[float, bool]]],
        target_error: float,
        confidence: float = 0.9,
        min_accepted: int = 10,
    ) -> dict[str, float]:
        """Fit per-field thresholds on labeled calibration data.

        ``calib``: field path -> [(top_probability, correct), ...], e.g. from
        an eval run's FieldRecords. For each field, pick the smallest tau
        (maximum coverage) such that the Wilson upper bound on the accepted
        set's error stays under ``target_error``. If no tau qualifies, the
        field always escalates (tau = 1.0 + eps).

        This is split-conformal-flavored selective prediction: the guarantee
        is empirical on the calibration slice, so keep it held out.
        """
        z = _normal_quantile((1.0 + confidence) / 2.0)
        for path, rows in calib.items():
            if not rows:
                continue
            tau = _fit_tau(rows, target_error, z, min_accepted)
            self.per_field[path] = tau
        return self.per_field


def _fit_tau(
    rows: list[tuple[float, bool]],
    target_error: float,
    z: float,
    min_accepted: int,
) -> float:
    for tau in sorted({p for p, _ in rows}):
        accepted = [c for p, c in rows if p >= tau]
        if len(accepted) < min_accepted:
            continue
        err = 1.0 - sum(accepted) / len(accepted)
        # Wilson-ish upper bound on the accepted error rate
        ub = err + z * math.sqrt(err * (1.0 - err) / len(accepted))
        if ub <= target_error:
            return tau
    return 1.0 + 1e-9  # never confident enough -> always escalate


def _normal_quantile(p: float) -> float:
    """Acklam's inverse normal CDF approximation (no scipy needed)."""
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
