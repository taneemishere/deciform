"""Escalation experiment: elephant decides; fallback LLM fills low-confidence
fields; compare combined accuracy/cost against pure elephant and pure mammoth.

    python scripts/escalation_experiment.py --task fintech --calib 120 --test 120

Phases:
  A. elephant on calib slice (accept-all) -> EscalationPolicy.fit(target_error)
  B. elephant + mammoth on test slice -> per-(example, field) records
  C. LLM produce() on every labeled field of the test slice -> fallback values
  D. offline sweep over tau -> accuracy vs escalation-fraction Pareto points,
     fitted-policy operating point, saved to results/escalation_{task}.json|png
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deciform import EscalationPolicy, Deciform
from deciform.backends.llm import OpenAIGenerativeBackend
from deciform.backends.systemone import SystemOneBackend
from deciform.compiler import compile
from deciform.core import _leaf_schema
from deciform.eval.tasks import TASKS

# any OpenAI-compatible chat endpoint for the fallback side
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def _get(tree, path):
    for part in path.split("."):
        if tree is None:
            return None
        tree = tree.get(part) if isinstance(tree, dict) else getattr(tree, part, None)
    return tree


def _match(val, label) -> bool:
    if isinstance(label, list) and isinstance(val, list):
        return set(map(str, val)) == set(map(str, label))
    return val == label


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fintech", choices=list(TASKS))
    p.add_argument("--calib", type=int, default=120)
    p.add_argument("--test", type=int, default=120)
    p.add_argument("--target-error", type=float, default=0.10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--llm-model", default=LLM_MODEL)
    p.add_argument("--tag", default=None, help="output filename suffix")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    # key files live next to the repo root; env vars take precedence
    _key_dir = Path(__file__).resolve().parent.parent
    if not os.environ.get("BASETEN_API_KEY"):
        os.environ["BASETEN_API_KEY"] = (_key_dir / "baseten_key").read_text().strip()
    if not os.environ.get("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = (_key_dir / "nvapi_key").read_text().strip()

    task = TASKS[args.task](n=args.calib + args.test) if args.task == "fintech" else TASKS[args.task]()
    calib_ex = task.examples[: args.calib]
    test_ex = task.examples[args.calib: args.calib + args.test]
    program = compile(task.model)
    plans = {pl.path: pl for pl in program.fields}
    label_fields = sorted(test_ex[0].labels)

    elephant = Deciform(backend=SystemOneBackend(model=SystemOneBackend.ELEPHANT))
    mammoth = Deciform(backend=SystemOneBackend(model=SystemOneBackend.MAMMOTH))
    llm = OpenAIGenerativeBackend(
        base_url=LLM_BASE_URL, api_key=os.environ["LLM_API_KEY"],
        model=args.llm_model, requests_per_minute=30)

    def decide_all(sf, ex):
        res = sf.extract(task.model, ex.state)
        dump = res.value.model_dump()
        out = {}
        for tr in res.fields:
            if tr.path not in ex.labels:
                continue
            out[tr.path] = {
                "value": _get(dump, tr.path),
                "prob": tr.answer.top_probability if tr.answer else 0.0,
                "skipped": tr.skipped,
            }
        return out

    # ---- A: fit thresholds on calib ----
    print("A. elephant on calib ...", flush=True)
    calib_rows: dict[str, list[tuple[float, bool]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda ex: decide_all(elephant, ex), calib_ex))
    for ex, fields in zip(calib_ex, results):
        for path, r in fields.items():
            if r["skipped"]:
                correct = ex.labels.get(path) is None and r["value"] is None
            else:
                correct = _match(r["value"], ex.labels.get(path))
            calib_rows.setdefault(path, []).append((r["prob"], correct))
    policy = EscalationPolicy()
    fitted = policy.fit(calib_rows, target_error=args.target_error)
    print("   fitted:", {k: round(v, 3) for k, v in fitted.items()})

    # ---- B: elephant + mammoth on test ----
    print("B. elephant + mammoth on test ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        ele = list(pool.map(lambda ex: decide_all(elephant, ex), test_ex))
        mam = list(pool.map(lambda ex: decide_all(mammoth, ex), test_ex))

    # ---- C: LLM fallback values, batched per example (1 call each) ----
    print("C. LLM fallback pass (this is the slow bit) ...", flush=True)

    field_hints = {
        path: _leaf_schema(plans[path]) if path in plans else {}
        for path in label_fields
    }

    def fallback_all(i):
        """One LLM call producing every field of example i."""
        ex = test_ex[i]
        hints = json.dumps(field_hints)
        prompt = (
            "Fill in every field of a structured decision object.\n\n"
            f"State:\n{json.dumps(ex.state, indent=2)}\n\n"
            f"Field schemas:\n{hints}\n\n"
            'Respond with a JSON object {"fields": {"<field>": <value>, ...}} '
            "covering every field. Each value must match its schema."
        )
        last = None
        for attempt in range(6):
            try:
                ch = llm._client.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=384, temperature=0.0,
                )
                text = (ch["message"].get("content") or "").strip()
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    obj = json.loads(m.group(0))
                    return obj.get("fields", obj)
                return {"__parse_error__": text[:200]}
            except httpx.HTTPStatusError as e:
                last = e
                ra = e.response.headers.get("retry-after") if e.response is not None else None
                try:
                    wait = float(ra) if ra else 15.0 * (attempt + 1)
                except (ValueError, TypeError):
                    wait = 15.0 * (attempt + 1)
                time.sleep(min(wait, 120.0))
            except httpx.TransportError as e:
                last = e
                time.sleep(3.0 * (attempt + 1))
        raise last

    llm_vals: list[dict[str, object]] = [dict() for _ in test_ex]
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fallback_all, i): i for i in range(len(test_ex))}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                llm_vals[i] = fut.result()
            except Exception as e:
                llm_vals[i] = {p: f"__error__:{type(e).__name__}" for p in label_fields}
            done += 1
            if done % 10 == 0:
                print(f"   llm {done}/{len(test_ex)}", flush=True)

    # ---- D: offline sweep ----
    print("D. sweeping thresholds ...", flush=True)
    taus = [i / 100 for i in range(0, 101, 5)]
    curve = []
    for tau in taus:
        per_field = {pth: {"correct": 0, "n": 0, "esc": 0} for pth in label_fields}
        for i, ex in enumerate(test_ex):
            for path in label_fields:
                rec = ele[i].get(path)
                if rec is None:
                    continue
                # skipped fields carry their gate's confidence now; escalate
                # only when the model was unsure whether the field exists
                use_fallback = rec["prob"] < tau
                if use_fallback:
                    val = llm_vals[i].get(path)
                    per_field[path]["esc"] += 1
                else:
                    val = rec["value"]
                per_field[path]["n"] += 1
                per_field[path]["correct"] += int(_match(val, ex.labels.get(path)))
        n_all = sum(f["n"] for f in per_field.values())
        curve.append({
            "tau": tau,
            "accuracy": sum(f["correct"] for f in per_field.values()) / max(1, n_all),
            "escalation_rate": sum(f["esc"] for f in per_field.values()) / max(1, n_all),
            "per_field": {k: (v["correct"] / v["n"] if v["n"] else 0.0)
                          for k, v in per_field.items()},
        })

    def acc_of(decisions, i, path):
        r = decisions[i].get(path)
        return r["value"] if r else None

    base = {"elephant": 0.0, "mammoth": 0.0}
    n_all = 0
    for i, ex in enumerate(test_ex):
        for path in label_fields:
            ev, mv = acc_of(ele, i, path), acc_of(mam, i, path)
            el_r, ma_r = ele[i][path], mam[i][path]
            for name, r, v in (("elephant", el_r, ev), ("mammoth", ma_r, mv)):
                ok = _match(v, ex.labels.get(path)) if not r["skipped"] else \
                    (ex.labels.get(path) is None and v is None)
                base[name] += int(ok)
            n_all += 1
    for name in base:
        base[name] /= n_all

    print(f"\n  baseline  elephant={base['elephant']:.3f}  mammoth={base['mammoth']:.3f}")
    print("  tau | acc | esc_rate")
    for c in curve:
        print(f"  {c['tau']:.2f} | {c['accuracy']:.3f} | {c['escalation_rate']:.3f}")

    # fitted operating point
    op = {"tau": None, "accuracy": 0.0, "escalation_rate": 0.0}
    per = {pth: {"correct": 0, "n": 0, "esc": 0} for pth in label_fields}
    for i, ex in enumerate(test_ex):
        for path in label_fields:
            rec = ele[i].get(path)
            if rec is None:
                continue
            tau = policy.threshold_for(path)
            if rec["prob"] < tau:
                val = llm_vals[i].get(path)
                per[path]["esc"] += 1
            else:
                val = rec["value"]
            per[path]["n"] += 1
            per[path]["correct"] += int(_match(val, ex.labels.get(path)))
    n_f = sum(f["n"] for f in per.values())
    op["accuracy"] = sum(f["correct"] for f in per.values()) / max(1, n_f)
    op["escalation_rate"] = sum(f["esc"] for f in per.values()) / max(1, n_f)
    op["per_field"] = {k: (v["correct"] / v["n"] if v["n"] else 0.0) for k, v in per.items()}
    print(f"\n  fitted-policy point: acc={op['accuracy']:.3f} "
          f"esc={op['escalation_rate']:.3f} taus={ {k: round(v,3) for k,v in fitted.items()} }")

    llm_errors = sum(1 for row in llm_vals for v in row.values()
                     if isinstance(v, str) and v.startswith("__error__"))
    print(f"  llm fallback errors: {llm_errors}/{sum(len(r) for r in llm_vals)}")

    payload = {
        "task": args.task, "n_test": len(test_ex), "n_calib": len(calib_ex),
        "target_error": args.target_error, "fitted_taus": fitted,
        "llm_model": args.llm_model,
        "baselines": base, "curve": curve, "fitted_point": op,
        "llm_errors": llm_errors,
        "elephant_records": ele, "mammoth_records": mam,
        "fallback_values": llm_vals,
        "test_labels": [ex.labels for ex in test_ex],
    }
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tag = f"__{args.tag}" if args.tag else ""
    (out / f"escalation_{args.task}{tag}.json").write_text(json.dumps(payload, indent=2))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([c["escalation_rate"] for c in curve], [c["accuracy"] for c in curve],
            marker="o", label=f"elephant + {args.llm_model.split('/')[-1]} fallback")
    ax.axhline(base["elephant"], ls="--", c="gray", label=f"elephant-only ({base['elephant']:.3f})")
    ax.axhline(base["mammoth"], ls=":", c="green", label=f"mammoth-only ({base['mammoth']:.3f})")
    ax.scatter([op["escalation_rate"]], [op["accuracy"]], c="red", zorder=5,
               label=f"fitted policy ({op['accuracy']:.3f})")
    ax.set_xlabel("escalation rate (fraction of fields sent to fallback LLM)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"escalation Pareto — {args.task}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(out / f"escalation_{args.task}{tag}.png", dpi=150, bbox_inches="tight")
    print(f"  saved -> {out}/escalation_{args.task}{tag}.json + .png")


if __name__ == "__main__":
    main()
