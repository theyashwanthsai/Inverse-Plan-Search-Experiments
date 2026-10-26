"""Experiment 1.2: accuracy and speed of SIPS vs unbiased / oracle BIRL.

    uv run python experiments/exp1_2_accuracy.py dkg
    uv run python experiments/exp1_2_accuracy.py taxi
    uv run python experiments/exp1_2_accuracy.py blockwords
"""
import json
import sys

import numpy as np

from sips.agent import AgentParams
from common.paths import TABLES
from evaluation.datasets import domain_setup, make_dataset
from evaluation.evaluate import aggregate, format_table, run_birl, run_sips, timed_vi
from evaluation.paper import PAPER_1_2
from sips.smc import SIPSConfig


def main(name):
    domains, admissible, n_unbiased, n_oracle = domain_setup(name)
    params = AgentParams(heuristic=domains[0].default_heuristic) if hasattr(domains[0], "default_heuristic") else AgentParams()
    rng = np.random.default_rng(2020)
    problems, rejected = make_dataset(domains, admissible, params, rng)
    print(f"{name}: {len(problems)} problems on {len(domains)} instances "
          f"({rejected} agent samples rejected for not reaching the goal within 100 steps)", flush=True)

    rows = {"SIPS": [], "BIRL (unbiased)": [], "BIRL (oracle)": [], "BIRL (exact VI, not in paper)": []}
    for d in domains:
        mine = [p for p in problems if p.domain is d]
        vts = {"BIRL (unbiased)": timed_vi("unbiased", d, n_iter=n_unbiased, seed=1),
               "BIRL (oracle)": timed_vi("oracle", d, [p.states for p in mine], n_iter=n_oracle, seed=2),
               "BIRL (exact VI, not in paper)": timed_vi("exact", d)}
        for i, p in enumerate(mine):
            cfg = SIPSConfig(n_particles=10 * d.n_goals, resample_threshold=0.25, rejuvenate=False)
            r = run_sips(p, params, cfg, seed=[3, i, domains.index(d)])
            rows["SIPS"].append({**r, "kind": p.kind})
            for meth, (vt, c0) in vts.items():
                rows[meth].append({**run_birl(p, vt, c0), "kind": p.kind})
        print(f"  {d.name}: done ({len(mine)} problems; unbiased VI {vts['BIRL (unbiased)'][1]:.1f}s)", flush=True)

    results = {m: aggregate(r) for m, r in rows.items()}
    by_kind = {kind: {m: aggregate([x for x in r if x["kind"] == kind]) for m, r in rows.items()}
               for kind in ("optimal", "suboptimal")}
    table = format_table(results, PAPER_1_2.get(name))
    print(table)
    for kind, res in by_kind.items():
        print(f"\n{kind} trajectories only:\n" + format_table(res))
    (TABLES / f"exp1_2_{name}.json").write_text(json.dumps({"all": results, **by_kind, "rows": rows,
                                                                  "n_problems": len(problems), "rejected": rejected}, indent=1))
    (TABLES / f"exp1_2_{name}.md").write_text(
        f"# Experiment 1.2: {name} ({len(problems)} problems)\n\n{table}\n\n"
        + "\n\n".join(f"## {k} only\n\n{format_table(v)}" for k, v in by_kind.items()) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dkg")
