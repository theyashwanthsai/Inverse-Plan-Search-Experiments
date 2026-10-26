"""Diagnostic (not a paper table): Exp 1.2 methods on one larger DKG map (2.4e7 product states),
to test whether unbiased BIRL goes flat once the state space is large, as the paper reports.

    uv run python experiments/diag_large_dkg.py
"""
import numpy as np

from sips.agent import AgentParams
from envs.dkg.maps import large_map
from evaluation.evaluate import aggregate, format_table, run_birl, run_sips, timed_vi
from sips.smc import SIPSConfig
from common.paths import TABLES
from evaluation.datasets import make_dataset


def main():
    d = large_map()
    params = AgentParams()
    problems, rej = make_dataset([d], "maze", params, np.random.default_rng(2020))
    print(f"large map: {len(problems)} problems, product space {d.state_space_size():.2e}, {rej} rejected", flush=True)
    vts = {"BIRL (unbiased)": timed_vi("unbiased", d, n_iter=250_000, seed=1),
           "BIRL (oracle)": timed_vi("oracle", d, [p.states for p in problems], n_iter=10_000, seed=2)}
    rows = {"SIPS": [run_sips(p, params, SIPSConfig(n_particles=30), seed=[5, i]) for i, p in enumerate(problems)]}
    for m, (vt, c0) in vts.items():
        rows[m] = [run_birl(p, vt, c0) for p in problems]
    table = format_table({m: aggregate(r) for m, r in rows.items()})
    print(table)
    (TABLES / "diag_large_dkg.md").write_text(f"# Large DKG map diagnostic ({len(problems)} problems)\n\n{table}\n")


if __name__ == "__main__":
    main()
