"""Aggregate results/*.json into a markdown comparison table + coverage curves.

    python scripts/summarize.py --results results/

Writes results/SUMMARY.md and results/coverage_{task}__{field}.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    out_dir = Path(args.results)

    reports = []
    for f in sorted(out_dir.glob("*.json")):
        reports.append(json.loads(f.read_text()))

    # ---- markdown table: task.field rows, backend columns ----
    cells: dict[tuple[str, str], dict[str, dict]] = {}
    backends = []
    for rep in reports:
        b = rep["backend"].split(":")[-1].replace("baseten-", "").replace("-latest", "")
        if b not in backends:
            backends.append(b)
        for path, m in rep["field_metrics"].items():
            cells.setdefault((rep["task"], path), {})[b] = m

    lines = ["# deciform eval summary\n"]
    for (task, path), by_backend in cells.items():
        lines.append(f"\n## {task} → `{path}`\n")
        lines.append("| backend | acc | ece | brier | auarc | flip |")
        lines.append("|---|---|---|---|---|---|")
        for b in backends:
            m = by_backend.get(b)
            if not m:
                continue
            flip = f"{m['order_flip_rate']:.3f}" if "order_flip_rate" in m else "—"
            lines.append(
                f"| {b} | {m['accuracy']:.3f} | {m['ece']:.3f} | {m['brier']:.3f} "
                f"| {m['auarc']:.3f} | {flip} |"
            )

    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    # ---- coverage curves: selective accuracy vs coverage per backend ----
    for (task, path), by_backend in cells.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        for b, m in by_backend.items():
            curve = m.get("coverage_curve") or []
            xs = [c["coverage"] for c in curve]
            ys = [1.0 - c["error"] for c in curve]
            ax.plot(xs, ys, marker=".", label=b)
        ax.set_xlabel("coverage (fraction auto-accepted)")
        ax.set_ylabel("selective accuracy")
        ax.set_title(f"{task} → {path}")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3); ax.legend()
        safe = f"{task}__{path}".replace("/", "_").replace(".", "_")
        fig.savefig(out_dir / f"coverage_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {out_dir/'SUMMARY.md'} and {len(cells)} coverage plots")


if __name__ == "__main__":
    main()
