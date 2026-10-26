"""Experiment 4.1 (thesis Table 4.1, mDKG rows): CLIPS vs unimodal inverse planning, with two LLM
utterance scorers.

Methods:
  CLIPS (<scorer>)          actions + utterance
  language-only (<scorer>)  utterance only
  action-only               actions only (no LLM)
Scorers: a local Ollama model scoring log P(u | c) exactly by teacher forcing, and an OpenAI chat
model asked yes/no per command (log P(Yes) as a stand-in). See src/llms/.

Usage: uv run python experiments/exp4_1_mdkg.py [--ollama MODEL] [--openai MODEL] [--no-openai]
"""
import argparse
import json
import time

import numpy as np

from clips.assist import gold_options, precision_recall, simulate
from clips.inference import GoalInference
from clips.planner import JointPlanner
from clips.problems import PROBLEMS, make_map, observed_states
from common.paths import TABLES
from llms.ollama import OllamaScorer
from llms.openai_judge import OpenAIJudgeScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", default="llama3.1:latest")
    ap.add_argument("--openai", default="gpt-4o-mini")
    ap.add_argument("--no-openai", action="store_true")
    args = ap.parse_args()

    scorers = [OllamaScorer(args.ollama)] + ([] if args.no_openai else [OpenAIJudgeScorer(args.openai)])
    methods = [("action-only", None, True, False)]
    for sc in scorers:
        methods += [(f"CLIPS ({sc.name})", sc, True, True), (f"language-only ({sc.name})", sc, False, True)]

    planners: dict = {}
    rows = []
    for p in PROBLEMS:
        P = planners.setdefault(p.map, JointPlanner(make_map(p.map)))
        env = P.env
        states = observed_states(env, p)
        g_true = env.goal_names.index(p.true_goal)
        gold = gold_options(P, states[-1], g_true, p.true_cost, p.options)
        for name, sc, use_a, use_u in methods:
            t0 = time.perf_counter()
            b = GoalInference(P, scorer=sc, use_actions=use_a, use_utterance=use_u).run(
                states, p.human_actions, p.utterance)
            secs = time.perf_counter() - t0
            post_gc = b.hypothesis_posterior()
            ep = simulate(P, post_gc, states[-1], g_true, p.true_cost, p.options)
            prec, rec = precision_recall(ep.robot_options, gold)
            row = dict(problem=p.name, category=p.category, method=name,
                       p_true=float(b.goal_posterior()[g_true]),
                       posterior=dict(zip(env.goal_names, np.round(b.goal_posterior(), 3).tolist())),
                       precision=prec, recall=rec, gold=sorted(gold), chosen=sorted(ep.robot_options),
                       steps=ep.steps, human_cost=ep.human_cost, reached=ep.reached, fallback=ep.fallback,
                       seconds=secs)
            rows.append(row)
            print(f"{p.name:20s} {name:42s} P(g)={row['p_true']:.2f} prec={prec:.2f} rec={rec:.2f} "
                  f"steps={ep.steps} cost={ep.human_cost:.1f} chosen={row['chosen']} gold={row['gold']}", flush=True)

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "exp4_1_mdkg.json").write_text(json.dumps(rows, indent=1))
    ol = scorers[0]
    note = (f"Ollama teacher forcing: {ol.n_censored} of {ol.n_tokens} scored tokens were outside the top 20 "
            "and got the 20th log-probability as an upper bound (counts exclude cached scores).")
    print(note)
    write_table(rows, [m[0] for m in methods], scorers[0].name, note)


def write_table(rows, method_names, ref_scorer: str, note: str = ""):
    """Means over problems; plan length and human cost relative to CLIPS with the first scorer,
    per problem, then averaged (as in thesis Table 4.1)."""
    ref = {r["problem"]: r for r in rows if r["method"] == f"CLIPS ({ref_scorer})"}
    lines = ["# Experiment 4.1: goal assistance in multi-agent DKG", "",
             f"{len(ref)} problems. Relative length / cost are against CLIPS ({ref_scorer}). {note}", "",
             "| Method | P(g_true) | Assist. prec. | Assist. rec. | Rel. length | Rel. human cost | Goal reached |",
             "|---|---|---|---|---|---|---|"]
    for m in method_names:
        rs = [r for r in rows if r["method"] == m]
        rel_len = np.mean([r["steps"] / ref[r["problem"]]["steps"] for r in rs])
        rel_cost = np.mean([r["human_cost"] / ref[r["problem"]]["human_cost"] for r in rs])
        se = np.std([r["p_true"] for r in rs], ddof=1) / np.sqrt(len(rs))
        lines.append(f"| {m} | {np.mean([r['p_true'] for r in rs]):.2f} ({se:.2f}) | "
                     f"{np.mean([r['precision'] for r in rs]):.2f} | {np.mean([r['recall'] for r in rs]):.2f} | "
                     f"{rel_len:.2f} | {rel_cost:.2f} | {np.mean([r['reached'] for r in rs]):.2f} |")
    lines += ["", "*thesis (mDKG, 30 problems):* CLIPS 0.74, 0.97 / 0.97; language-only 0.55, 0.90 / 0.83, "
              "rel. length 1.26, rel. cost 1.18; action-only 0.31, 0.43 / 0.40, 1.68, 1.46.", "",
              "## Per problem: P(g_true)", "",
              "| problem | " + " | ".join(method_names) + " |", "|---|" + "---|" * len(method_names)]
    for prob in ref:
        vals = {r["method"]: r["p_true"] for r in rows if r["problem"] == prob}
        lines.append(f"| {prob} | " + " | ".join(f"{vals[m]:.2f}" for m in method_names) + " |")
    (TABLES / "exp4_1_mdkg.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
