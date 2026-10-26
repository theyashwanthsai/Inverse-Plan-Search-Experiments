import numpy as np
import pytest

from clips.assist import gold_options, precision_recall, qmdp_action, simulate
from clips.commands import command_distribution, lift, valid
from clips.env import HELD_H, HELD_R, ON_MAP, USED, MDKG
from clips.inference import BETAS, GoalInference, beta_prior
from clips.planner import COST_FNS, JointPlanner, boltzmann_logprobs
from clips.problems import PROBLEMS, make_map, observed_states
from llms.ollama import BPETokenizer, OllamaScorer

TINY = ["#######",
        "#1#2..#",
        "#R#...#",
        "#@.&.r#",
        "#######"]


@pytest.fixture(scope="module")
def tiny():
    return JointPlanner(MDKG(TINY, {"1": "red", "2": "yellow"}, "tiny"))


@pytest.fixture(scope="module")
def red_keys():
    return JointPlanner(make_map("red_keys"))


class FakeScorer:
    """Scores a command by a fixed table of substrings; no LLM."""
    name = "fake"

    def __init__(self, liked: dict):
        self.liked = liked

    def loglik(self, utterance, commands, examples):
        return np.array([max([v for k, v in self.liked.items() if k in c], default=-20.0) for c in commands])


# ---------------- environment ----------------
def test_keys_single_use_handover_and_no_robot_gems(tiny):
    env = tiny.env
    s = env.initial_state  # human at (3,1), robot at (3,3), key at (3,5), gem1 behind red door
    s = env.step(s, "wait")
    s = env.step(s, "right")  # robot (3,4)
    s = env.step(s, "wait")
    s = env.step(s, "right")  # robot (3,5) on the key
    s = env.step(s, "wait")
    assert "pickup" in env.legal_actions(s)
    s = env.step(s, "pickup")
    assert s[5] == (HELD_R,)
    assert not any(a.startswith("handover") for a in env.legal_actions(s))  # human too far away
    # Robot walks next to the human and hands the key over.
    for _ in range(3):
        s = env.step(s, "wait")
        s = env.step(s, "left")
    s = env.step(s, "wait")
    assert "handover_key1" in env.legal_actions(s)
    s = env.step(s, "handover_key1")
    assert s[5] == (HELD_H,)
    assert "unlock_up" in env.legal_actions(s)
    s = env.step(s, "unlock_up")
    assert s[5] == (USED,) and s[6] == 0
    # The human walks through the opened door onto the gem and picks it up.
    s = env.step(s, "wait")  # robot
    s = env.step(s, "up")  # human onto the door cell
    s = env.step(s, "wait")
    s = env.step(s, "up")  # human onto the gem
    s = env.step(s, "wait")
    assert "pickup" in env.legal_actions(s) and env.step(s, "pickup")[7] == 0


def test_robot_never_picks_up_gems(red_keys):
    env = red_keys.env
    for i in range(red_keys.n_states):
        s = red_keys.states[i]
        if s[0] == 1 and (s[3], s[4]) in env.gem_at:
            k = env.key_at.get((s[3], s[4]))
            assert "pickup" not in env.legal_actions(s) or (k is not None and s[5][k] == ON_MAP)


# ---------------- planner ----------------
def test_no_duplicate_edges(red_keys):
    pairs = red_keys.src * red_keys.n_states + red_keys.dst
    assert len(np.unique(pairs)) == len(pairs)


@pytest.mark.parametrize("c", range(len(COST_FNS)))
def test_value_function_satisfies_bellman_optimality(red_keys, c):
    rng = np.random.default_rng(0)
    for g in range(red_keys.env.n_goals):
        V = red_keys.V(g, c)
        for i in rng.choice(red_keys.n_states, 300, replace=False):
            s = red_keys.states[i]
            if s[7] >= 0:
                assert V[i] == (0 if s[7] == g else np.inf)
                continue
            _, q = red_keys.Q(s, g, c)
            assert np.isclose(V[i], q.min()) or (np.isinf(V[i]) and np.isinf(q.min()))


def test_boltzmann_logprobs():
    q = np.array([3.0, 4.0, np.inf])
    lp = boltzmann_logprobs(q, np.array([0.5, 2.0]))
    assert np.allclose(np.exp(lp).sum(axis=1), 1) and np.all(np.isneginf(lp[:, 2]))
    assert lp[1, 0] - lp[1, 1] == pytest.approx(2.0)
    assert np.allclose(np.exp(boltzmann_logprobs(np.full(3, np.inf), np.array([1.0]))), 1 / 3)


# ---------------- utterance model ----------------
def test_command_enumeration_pruning_and_lifting(red_keys):
    env = red_keys.env
    sal = [("pickup", "robot", "key1"), ("handover", "robot", "key1"), ("unlock", "human", "door1")]
    assert not valid([sal[2]])  # speaker-only
    assert not valid(sal[:2])  # dependent chain (same key)
    assert lift(env, [sal[1]]) == "(handover you me ?key1) where (iscolor ?key1 red)"
    dist = command_distribution(env, sal, K=3)
    assert sum(dist.values()) == pytest.approx(1)
    assert set(dist) == {
        "(pickup you ?key1) where (iscolor ?key1 red)",
        "(handover you me ?key1) where (iscolor ?key1 red)",
        "(pickup you ?key1) (unlock me ?door1) where (iscolor ?key1 red) (iscolor ?door1 red)",
        "(handover you me ?key1) (unlock me ?door1) where (iscolor ?key1 red) (iscolor ?door1 red)",
    }
    # Different keys of the same colour lift to the same command, and their mass adds up.
    two = command_distribution(env, [("pickup", "robot", "key1"), ("pickup", "robot", "key2")], K=1)
    assert two == {"(pickup you ?key1) where (iscolor ?key1 red)": 1.0}


# ---------------- inference ----------------
def test_inference_matches_brute_force(red_keys):
    p = PROBLEMS[0]
    states = observed_states(red_keys.env, p)
    scorer = FakeScorer({"(handover you me": -1.0, "?key1 red": -2.0})
    b = GoalInference(red_keys, scorer=scorer).run(states, p.human_actions, p.utterance)
    inf = GoalInference(red_keys, scorer=scorer)
    G, C = red_keys.env.n_goals, len(COST_FNS)
    logw = np.zeros((G, C, len(BETAS))) + np.log(beta_prior())
    human = [s for s in states[:-1] if s[0] == 0]
    for g in range(G):
        for c in range(C):
            for bi, beta in enumerate(BETAS):
                for s, a in zip(human, p.human_actions):
                    acts, q = red_keys.Q(s, g, c)
                    w = np.exp(-beta * (q - q[np.isfinite(q)].min()))
                    logw[g, c, bi] += np.log(w[acts.index(a)] / w.sum())
            dist = inf.utterance_model(states[-1])[(g, c)]
            lik = sum(np.exp(scorer.loglik("", [cmd], [])[0]) * m for cmd, m in dist.items())
            logw[g, c] += np.log(lik) if lik > 0 else -np.inf
    w = np.exp(logw - logw[np.isfinite(logw)].max()).sum(axis=(1, 2))
    assert np.allclose(b.goal_posterior(), w / w.sum())


def test_robot_actions_are_not_evidence(red_keys):
    inf = GoalInference(red_keys)
    b = inf.init()
    robot_turn = red_keys.env.step(red_keys.env.initial_state, "left")
    with pytest.raises(AssertionError):
        inf.observe_action(b, robot_turn, "wait")


def test_language_only_follows_the_utterance(red_keys):
    # Only commands about the blue door/key are liked: only the green gem's plan has one.
    scorer = FakeScorer({"(iscolor ?door1 blue)": 0.0, "(iscolor ?key1 blue)": 0.0})
    s = red_keys.env.initial_state
    b = GoalInference(red_keys, scorer=scorer, use_actions=False).run([s], (), "open the blue door")
    assert red_keys.env.goal_names[int(np.argmax(b.goal_posterior()))] == "green"


# ---------------- assistance ----------------
def test_qmdp_fetches_the_right_red_key(red_keys):
    env = red_keys.env
    for goal, key in (("red", "key1"), ("blue", "key2")):
        g = env.goal_names.index(goal)
        post = np.zeros((env.n_goals, len(COST_FNS)))
        post[g, 2] = 1.0
        s = env.step(env.initial_state, "wait")  # robot's turn
        assert gold_options(red_keys, s, g, 2, "keys") == {key}
        ep = simulate(red_keys, post, s, g, 2, "keys")
        assert ep.reached and ep.robot_options == {key} and not ep.fallback
        assert qmdp_action(red_keys, post, s) in env.legal_actions(s)


def test_precision_recall():
    assert precision_recall({"key1"}, {"key1", "key2"}) == (1.0, 0.5)
    assert precision_recall(set(), {"key1"}) == (0.0, 0.0)
    assert precision_recall({"key3"}, {"key1"}) == (0.0, 0.0)


# ---------------- LLM scorer plumbing (no network) ----------------
TOY_MERGES = ["c a", "ca n", "Ġ y", "Ġy o", "Ġyo u"]  # Ġ is the byte-level space


def test_bpe_tokenizer_on_toy_merges():
    tok = BPETokenizer(TOY_MERGES)
    assert tok.encode("can you") == ["can", " you"]
    assert tok.encode("cab?") == ["ca", "b", "?"]


def test_ollama_teacher_forcing_censors_tokens_outside_top20(monkeypatch):
    sc = OllamaScorer("fake")
    sc._tok = BPETokenizer(TOY_MERGES)
    prompts = []

    def fake_top20(prompt):
        prompts.append(prompt)
        return {"can": -1.0, " you": -2.0, "zz": -7.0}

    monkeypatch.setattr(sc, "_top20", fake_top20)
    # Tokens: "can", " you", and the closing quote, which is not in the top 20 -> bound -7.
    assert sc.score_one("can you", "(cmd)", []) == pytest.approx(-10.0)
    assert (sc.n_tokens, sc.n_censored) == (3, 1)
    assert prompts[1].endswith('Utterance: "can') and prompts[2].endswith('Utterance: "can you')


# ---------------- baselines: literal listener, verbal scorer, vision ----------------
from clips.literal import LiteralListener, literal_commands, plan_command, systematic_sample  # noqa: E402
from clips.render import render_png  # noqa: E402
from llms.openai_vision import build_prompt  # noqa: E402

RED_PICKUP = "(pickup you ?key1) where (iscolor ?key1 red)"


def test_literal_commands_are_assistant_only_and_grouped(red_keys):
    cmds = literal_commands(red_keys.env, red_keys.env.initial_state)
    assert all(" me " not in c.split(" where ")[0] or "(handover you me" in c for c in cmds)
    assert sorted(cmds[RED_PICKUP]) == [(("pickup", "robot", "key1"),), (("pickup", "robot", "key2"),)]


def test_systematic_sampling_matches_probabilities():
    picks = systematic_sample(np.array([0.2, 0.7, 0.1]), 10, np.random.default_rng(0))
    assert np.bincount(picks, minlength=3).tolist() == [2, 7, 1]


def test_plan_command_carries_out_the_command(tiny):
    path, cost = plan_command(tiny, tiny.env.initial_state, (("handover", "robot", "key1"),), COST_FNS[0].table())
    assert path[-1][1] == "handover_key1" and np.isfinite(cost)
    assert tiny.env.step(*path[-1])[5] == (HELD_H,)


def test_stay_mode_lets_the_robot_only_wait_or_hand_over(red_keys):
    s = red_keys.env.step(red_keys.env.initial_state, "wait")  # robot's turn
    acts, q = red_keys.Q(s, 0, 0, "stay")
    assert all(np.isinf(v) for a, v in zip(acts, q) if a != "wait" and not a.startswith("handover"))


def test_literal_naive_vs_efficient_grounding(red_keys):
    # Only "pick up a red key" is liked. Efficient takes the red key nearest the robot (key2);
    # naive averages over both red keys.
    env = red_keys.env
    s = env.step(env.initial_state, "wait")
    g = env.goal_names.index("red")
    r = LiteralListener(red_keys, FakeScorer({RED_PICKUP: 0.0})).assist(s, "", g, 2, "keys", {"key1"})
    assert r["top_command"] == RED_PICKUP
    assert r["efficient"]["precision"] == 0.0 and r["naive"]["precision"] == pytest.approx(0.5)
    assert r["naive"]["reached"] == 1.0  # the robot hands the wrong key over or the human fetches key1


def test_vision_stimulus_and_prompt(red_keys):
    png = render_png(red_keys.env, red_keys.env.initial_state, [(3, 4), (3, 3)])
    assert png[:4] == b"\x89PNG"
    assert "left, left" in build_prompt(["left", "left"], "Hi", "keys") and "door(s)" in build_prompt([], "Hi", "doors")


# ---------------- decision models (no network) ----------------
from clips.render import describe_text, key_descriptions  # noqa: E402
from llms.decisions import DecisionScorer, e2e_state  # noqa: E402


class FakeDecider:
    name = "fake-decider"

    def __init__(self):
        self.calls = []

    def ask(self, state, yesno, choices={}):
        self.calls.append(yesno)
        return {n: (0.9 if "?key1 red" in t else 0.0) for n, t in yesno.items()}


def test_decision_scorer_batches_and_floors(tmp_path, monkeypatch):
    monkeypatch.setattr("llms.base.CACHE_DIR", tmp_path)
    d = FakeDecider()
    sc = DecisionScorer(d)
    out = sc.loglik("get the red key", [RED_PICKUP, "(pickup you ?key1) where (iscolor ?key1 blue)"], [])
    assert out[0] == pytest.approx(np.log(0.9)) and out[1] == pytest.approx(np.log(1e-3))
    assert len(d.calls) == 1 and len(d.calls[0]) == 2  # one request for all commands


def test_text_description_names_every_object(red_keys):
    s = red_keys.env.initial_state
    text = e2e_state(describe_text(red_keys.env, s), [], "Hi")
    assert "key2: red key at row 5, column 6." in text and "door3: blue door" in text
    assert set(key_descriptions(red_keys.env, s)) == {"key1", "key2", "key3"}
