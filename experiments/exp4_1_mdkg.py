"""Experiment 4.1 (thesis Table 4.1, mDKG rows): CLIPS vs unimodal inverse planning, literal
listeners and a multimodal LLM, with three LLM utterance scorers.

Methods:
  CLIPS (<scorer>)              actions + utterance
  language-only (<scorer>)      utterance only
  action-only                   actions only (no LLM)
  literal naive / efficient (<scorer>)   the instruction alone, carried out literally
  vision (<model>)              a multimodal LLM given the rules, an image, the moves and the instruction
  decision-e2e (<model>)        a decision model given the rules, the map as text, the moves and the
                                instruction; returns P(goal) and P(help) per key or door
Scorers: a local Ollama model scoring log P(u | c) exactly by teacher forcing; an OpenAI chat model
asked yes/no per command (log P(Yes)); the same model writing its own probability as structured
output; and two decision models (Jev via OpenRouter, OpenAI Decisions) giving P(yes) per command.
See src/llms/.

Usage: uv run python experiments/exp4_1_mdkg.py [--ollama MODEL] [--openai MODEL] [--vision MODEL]
                                               [--no-openai] [--no-jev]
"""
import argparse
import json
import time

import numpy as np

from clips.assist import gold_options, precision_recall, simulate
from clips.env import ROBOT
from clips.inference import GoalInference
from clips.literal import LiteralListener
from clips.planner import JointPlanner
from clips.problems import PROBLEMS, make_map, observed_states
from clips.render import describe_text, door_descriptions, key_descriptions, render_png
from common.paths import TABLES
from llms.decisions import DecisionBaseline, DecisionScorer, Jev, OpenAIDecisions, e2e_state
from llms.ollama import OllamaScorer
from llms.openai_judge import OpenAIJudgeScorer
from llms.openai_verbal import OpenAIVerbalScorer
from llms.openai_vision import VisionBaseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", default="llama3.1:latest")
    ap.add_argument("--openai", default="gpt-4o-mini")
    ap.add_argument("--vision", default="gpt-5.5")
    ap.add_argument("--decisions", default="gpt-6-luna")
    ap.add_argument("--no-openai", action="store_true")
    ap.add_argument("--no-jev", action="store_true")
    args = ap.parse_args()

    ollama = OllamaScorer(args.ollama)
    judge = None if args.no_openai else OpenAIJudgeScorer(args.openai)
    scorers = [ollama] + ([] if args.no_openai else [judge, OpenAIVerbalScorer(args.openai)])
    literal_scorers = [ollama] + ([] if args.no_openai else [judge])
    vision = None if args.no_openai else VisionBaseline(args.vision)
    deciders = ([] if args.no_jev else [Jev()]) + ([] if args.no_openai else [OpenAIDecisions(args.decisions)])
    scorers += [DecisionScorer(d) for d in deciders]
    e2e = [DecisionBaseline(d) for d in deciders]

    methods = [("action-only", None, True, False)]
    for sc in scorers:
        methods += [(f"CLIPS ({sc.name})", sc, True, True), (f"language-only ({sc.name})", sc, False, True)]
    names = [m[0] for m in methods]
    names += [f"literal {v} ({sc.name})" for sc in literal_scorers for v in ("naive", "efficient")]
    names += [vision.name] if vision else []
    names += [b.name for b in e2e]

    planners: dict = {}
    rows = []

    def add(row):
        rows.append(row)
        pt = "  –  " if row["p_true"] is None else f"{row['p_true']:.2f}"
        print(f"{row['problem']:20s} {row['method']:50s} P(g)={pt} prec={row['precision']:.2f} rec={row['recall']:.2f} "
              f"steps={row['steps']} detail={row.get('chosen', '')}", flush=True)

    for p in PROBLEMS:
        P = planners.setdefault(p.map, JointPlanner(make_map(p.map)))
        env = P.env
        states = observed_states(env, p)
        g_true = env.goal_names.index(p.true_goal)
        gold = gold_options(P, states[-1], g_true, p.true_cost, p.options)
        base = dict(problem=p.name, category=p.category, gold=sorted(gold))

        for name, sc, use_a, use_u in methods:
            t0 = time.perf_counter()
            b = GoalInference(P, scorer=sc, use_actions=use_a, use_utterance=use_u).run(
                states, p.human_actions, p.utterance)
            secs = time.perf_counter() - t0
            ep = simulate(P, b.hypothesis_posterior(), states[-1], g_true, p.true_cost, p.options)
            prec, rec = precision_recall(ep.robot_options, gold)
            add(base | dict(method=name, p_true=float(b.goal_posterior()[g_true]),
                            posterior=dict(zip(env.goal_names, np.round(b.goal_posterior(), 3).tolist())),
                            precision=prec, recall=rec, chosen=sorted(ep.robot_options), steps=ep.steps,
                            human_cost=ep.human_cost, reached=ep.reached, fallback=ep.fallback, seconds=secs))

        utterance = p.literal_utterance or p.utterance
        for sc in literal_scorers:
            r = LiteralListener(P, sc).assist(states[-1], utterance, g_true, p.true_cost, p.options, gold)
            for v in ("naive", "efficient"):
                add(base | dict(method=f"literal {v} ({sc.name})", p_true=None, precision=r[v]["precision"],
                                recall=r[v]["recall"], steps=r[v]["steps"], human_cost=r[v]["human_cost"],
                                reached=r[v]["reached"], fallback=None,
                                chosen=f"top command {r['top_command']} ({r['top_prob']:.2f} of {r['n_commands']})"))

        if vision:
            trail = [(env.initial_state[1], env.initial_state[2])] + [(s[1], s[2]) for s in states if s[0] == ROBOT]
            a = vision.ask(render_png(env, states[-1], trail), list(p.human_actions), p.utterance, p.options)
            prec, rec = precision_recall(set(a.assist), gold)
            add(base | dict(method=vision.name, p_true=float(a.goal == p.true_goal), precision=prec, recall=rec,
                            chosen=sorted(a.assist), answer_goal=a.goal, steps=None, human_cost=None,
                            reached=None, fallback=None))

        state_text = e2e_state(describe_text(env, states[-1]), list(p.human_actions), p.utterance)
        cands = (key_descriptions if p.options == "keys" else door_descriptions)(env, states[-1])
        for b in e2e:
            a = b.ask(state_text, env.goal_names, cands, p.options)
            chosen = {o for o in cands if a[o] > 0.5}
            prec, rec = precision_recall(chosen, gold)
            add(base | dict(method=b.name, p_true=a["goal"][p.true_goal], posterior=a["goal"], precision=prec,
                            recall=rec, chosen=sorted(chosen), help_probs={o: a[o] for o in cands}, steps=None,
                            human_cost=None, reached=None, fallback=None))

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "exp4_1_mdkg.json").write_text(json.dumps(rows, indent=1))
    note = (f"Ollama teacher forcing: {ollama.n_censored} of {ollama.n_tokens} scored tokens were outside the top 20 "
            "and got the 20th log-probability as an upper bound (counts exclude cached scores).")
    print(note)
    write_table(rows, names, ollama.name, note)


def _mean(rs, f):
    vals = [r[f] for r in rs if r[f] is not None]
    return np.mean(vals) if vals else None


def write_table(rows, method_names, ref_scorer: str, note: str = ""):
    """Means over problems; plan length and human cost relative to CLIPS with the first scorer,
    per problem, then averaged (as in thesis Table 4.1). '–' where a method has no such number."""
    ref = {r["problem"]: r for r in rows if r["method"] == f"CLIPS ({ref_scorer})"}
    fmt = lambda v: "–" if v is None else f"{v:.2f}"
    lines = ["# Experiment 4.1: goal assistance in multi-agent DKG", "",
             f"{len(ref)} problems. Relative length / cost are against CLIPS ({ref_scorer}). {note}", "",
             "Literal listeners: mean over 10 systematically sampled commands; they infer no goal, so P(g_true) is '–'. "
             "Vision: P(g_true) is 1 if the single goal it names is the true one; it gives no plan, so length and cost are '–'. "
             "Decision e2e: P(g_true) is the model's own goal probability; it helps with every key or door it gives P > 0.5; no plan.", "",
             "| Method | P(g_true) | Assist. prec. | Assist. rec. | Rel. length | Rel. human cost | Goal reached |",
             "|---|---|---|---|---|---|---|"]
    for m in method_names:
        rs = [r for r in rows if r["method"] == m]
        pt = [r["p_true"] for r in rs if r["p_true"] is not None]
        p_cell = f"{np.mean(pt):.2f} ({np.std(pt, ddof=1) / np.sqrt(len(pt)):.2f})" if pt else "–"
        has_len = [r for r in rs if r["steps"] is not None]
        rel_len = np.mean([r["steps"] / ref[r["problem"]]["steps"] for r in has_len]) if has_len else None
        rel_cost = np.mean([r["human_cost"] / ref[r["problem"]]["human_cost"] for r in has_len]) if has_len else None
        lines.append(f"| {m} | {p_cell} | {fmt(_mean(rs, 'precision'))} | {fmt(_mean(rs, 'recall'))} | "
                     f"{fmt(rel_len)} | {fmt(rel_cost)} | {fmt(_mean(rs, 'reached'))} |")
    with_p = [m for m in method_names if any(r["method"] == m and r["p_true"] is not None for r in rows)]
    lines += ["", "*thesis (mDKG, 30 problems):* CLIPS 0.74, 0.97 / 0.97; language-only 0.55, 0.90 / 0.83, "
              "rel. length 1.26, rel. cost 1.18; action-only 0.31, 0.43 / 0.40, 1.68, 1.46; "
              "literal efficient –, 0.65 / 0.54, 1.55, 1.58; literal naive –, 0.58 / 0.47, 1.54, 1.52; "
              "GPT-4V 0.29, 0.39 / 0.37.", "",
              "## Per problem: P(g_true)", "",
              "| problem | " + " | ".join(with_p) + " |", "|---|" + "---|" * len(with_p)]
    for prob in ref:
        vals = {r["method"]: r["p_true"] for r in rows if r["problem"] == prob}
        lines.append(f"| {prob} | " + " | ".join(fmt(vals.get(m)) for m in with_p) + " |")
    lines += ["", "## Per problem: assistance recall", "",
              "| problem | " + " | ".join(method_names) + " |", "|---|" + "---|" * len(method_names)]
    for prob in ref:
        vals = {r["method"]: r["recall"] for r in rows if r["problem"] == prob}
        lines.append(f"| {prob} | " + " | ".join(fmt(vals.get(m)) for m in method_names) + " |")
    (TABLES / "exp4_1_mdkg.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
