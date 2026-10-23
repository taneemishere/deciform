"""Run eval tasks across backends -> JSON reports in results/.

    python scripts/run_eval.py \
        --tasks banking77 rte fintech \
        --backends elephant mammoth nim-jev \
        --limit 200 --workers 8 --out results/

`--flip` adds the order-reversal probe (choice fields, +1 call/example).
`--fit-error` fits EscalationPolicy per-field thresholds on the run's
calibration rows and prints them.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deciform import EscalationPolicy
from deciform.backends.llm import LLMAsJevBackend
from deciform.backends.systemone import SystemOneBackend
from deciform.eval.runner import run_task
from deciform.eval.tasks import TASKS

# any OpenAI-compatible chat endpoint for the LLM-as-Jev / fallback side
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

BACKENDS = {
    "elephant": lambda: SystemOneBackend(model=SystemOneBackend.ELEPHANT),
    "mammoth": lambda: SystemOneBackend(model=SystemOneBackend.MAMMOTH),
    "llm-jev": lambda: LLMAsJevBackend(
        base_url=LLM_BASE_URL, api_key=os.environ["LLM_API_KEY"],
        model=LLM_MODEL, requests_per_minute=30),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", required=True, choices=list(TASKS))
    p.add_argument("--backends", nargs="+", required=True, choices=list(BACKENDS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--flip", action="store_true", help="order-reversal probe")
    p.add_argument("--fit-error", type=float, default=None,
                   help="target error for EscalationPolicy.fit")
    p.add_argument("--gate-threshold", type=float, default=0.5,
                   help="presence probability at which Optional gates fire")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for task_name in args.tasks:
        loader = TASKS[task_name]
        task = loader(limit=args.limit) if task_name != "fintech" else loader(n=args.limit or 300)
        print(f"\n### {task.name}: {task.description}")
        for bname in args.backends:
            backend = BACKENDS[bname]()
            print(f"--- {backend.name} ---", flush=True)
            report = run_task(task, backend, workers=args.workers,
                              check_order_flip=args.flip,
                              gate_threshold=args.gate_threshold)
            for path, m in report.field_metrics.items():
                print(
                    f"  {path:18s} acc={m['accuracy']:.3f} ece={m['ece']:.3f} "
                    f"brier={m['brier']:.3f} auarc={m['auarc']:.3f}"
                    + (f" flip={m['order_flip_rate']:.3f}" if "order_flip_rate" in m else "")
                    + (f" gateP={m['gate_precision']:.2f} gateR={m['gate_recall']:.2f}"
                       if m.get("gate_recall") is not None else "")
                )
            print(f"  errors={report.n_errors} latency_p50={report.example_latency['p50']:.0f}ms "
                  f"p95={report.example_latency['p95']:.0f}ms wall={report.wall_seconds:.0f}s")

            out_path = Path(args.out) / f"{task.name}__{bname}.json"
            report.save(str(out_path))
            print(f"  saved -> {out_path}")

            if args.fit_error is not None:
                policy = EscalationPolicy()
                fitted = policy.fit(report.calibration_rows(), target_error=args.fit_error)
                print(f"  fitted thresholds (target_error={args.fit_error}): "
                      f"{json.dumps({k: round(v, 3) for k, v in fitted.items()})}")


if __name__ == "__main__":
    main()
