# Reproducing SIPS and CLIPS

A from-scratch Python reproduction of **SIPS** (Sequential Inverse Plan Search) and **CLIPS** (Cooperative Language-guided Inverse Plan Search), from this [talk](https://youtu.be/5rQFnHnd4ag?si=DBcKsDABIk5tEea-) and this [thesis](https://ztangent.github.io/assets/pdf/2025-phd-thesis-scaling-cooperative-intelligence.pdf) *Scaling Cooperative Intelligence via Inverse Planning and Probabilistic Programming* (MIT, 2025) by Prof. Tan Zhi-Xuan. I tried building everything from the thesis text alone, without reading the original codebase. No probabilistic-programming library is used: the particle filter, the planners, and the exact-inference checks are written by hand in raw python.

Most of this README is about CLIPS. I re-implemented it, ran it against the thesis's baselines, and then tried a few ideas of my own about how the language model should be used inside it, mostly because the OpenAI models the thesis relied on are gone.

---

## SIPS: goal inference from actions

SIPS infers what an agent wants by watching what it does, assuming the agent is a *boundedly rational planner*: it makes partial plans with limited search, replans when they run out, and sometimes acts noisily. Inference is a particle filter over (goal, current plan).

```
                      ┌──────────────────────────────────────────────┐
  goal prior ───────► │  particles: (goal g, partial plan, budget)   │
                      └──────────────────────┬───────────────────────┘
                                             │  for each observed action a_t
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │  extend   : if the plan ran out, replan with │
                      │             probabilistic A* (random budget) │
                      │  weight   : w *= P(a_t | plan) (ε-noisy)     │
                      │  resample : when the effective size drops    │
                      │  rejuvenate: MH moves on goal and on plan    │
                      └──────────────────────┬───────────────────────┘
                                             ▼
                               P(goal | a_1..t), updated online
```

The thesis compares SIPS with Bayesian inverse reinforcement learning (BIRL) on Doors-Keys-Gems, Taxi and Block Words  (I belive they can still be made better, but thats something I would take up in the future). I reproduced those experiments (thesis Exp. 1.1–1.3) and checked that SIPS converges to the exact posterior on a small map (error falls like 1/√particles). I also implemented the Ch. 2 extension (plan and action mistakes) and the Ch. 3 extension (open-ended goals over 458 words). 

The headline results held: SIPS beats unbiased BIRL in accuracy and cost on large DKG maps, and accuracy drops mainly for very noisy agents. My absolute numbers differ from the paper's, because the maps and datasets are my own.

## CLIPS: goal inference from actions *and* language

CLIPS extends inverse planning to a human and a robot working together. The human moves, then says something ambiguous ("Can you get me the red key?" when there are two red keys). The robot has to work out the goal from both clues, then help.

```
  human's moves ──► planner ──► P(moves | goal, cost, β)  ───────────┐
                    (Boltzmann over exact Q-values)                   │
                                                                      ▼
  human's sentence ─┐                                           ┌───────────┐
                    │   for each hypothesis (goal, cost):       │  Bayes:   │
                    │   roll out the joint plan                 │  multiply │──► P(goal)
                    │     → salient actions (pickup / unlock /  │  and sum  │
                    │       handover)                           │  over β   │
                    │     → candidate commands, lifted:         └─────┬─────┘
                    │       (pickup you ?key1)                        │
                    │       where (iscolor ?key1 red)                 ▼
                    ▼                                           ┌───────────┐
              LLM scores the sentence against each command      │  Q_MDP:   │
              → P(sentence | command)  ─────────────────────►   │  robot    │──► robot acts
                                                                │  helps    │
                                                                └───────────┘
```

The LLM never sees the map or the goals. It only answers one narrow question: *does this sentence go with this command?* The planner turns that into evidence about goals. Hence why, `language guided inverse plan search`.
For example, the red and blue gems both need "a red key", so the sentence cannot separate them and the moves have to.

My implementation:
- **Inference:** exact enumeration over 4 goals × 4 cost functions × 33 rationality values β, with a Gamma(0.5, 1) prior on β. The robot's own actions are interventions and are never conditioned on.
- **Planning:** exact V\* by Dijkstra over the full joint state space (72k–290k states), where the thesis used real-time A\*.
- **Commands:** commands are generated with the thesis's pruning rules (K = 3) and lifted to typed variables plus colours.
- **Assistance:** Q_MDP over the posterior frozen after the observations ("offline" mode), against a simulated human who gives up on an unhelpful robot.
- **Problems:** 10 goal-assistance problems of my own on two maps, covering the thesis's problem types (ambiguous, partial and joint instructions; low and high goal uncertainty). The thesis states 30 problems, but I did not have the bandwidth for that.

## Results (Experiment 4.1, multi-agent Doors-Keys-Gems)

All methods on the same 10 problems. P(g_true) is the probability on the true goal. Top-1 is 1 if the most probable goal is the true one, and 0.5 for a two-way tie. Precision and recall are over the keys or doors the robot picks up or unlocks, against the optimal plan for the true goal. Plan length is relative to CLIPS with llama3.1. A dash means the method doesn't produce that number.

| Method | Language model | P(g_true) | Top-1 | Prec. | Rec. | Rel. length |
|---|---|---|---|---|---|---|
| **CLIPS** | llama3.1 8B (Ollama), log P(u \| c) | 0.79 | 0.85 | 0.80 | 0.80 | 1.00 |
| **CLIPS** | gpt-4o-mini, yes/no judge | 0.81 | 0.85 | 0.90 | 0.90 | 1.07 |
| **CLIPS** | gpt-4o-mini, verbal probability | 0.64 | 0.85 | 0.45 | 0.50 | 1.71 |
| **CLIPS** | Jev (decision model) | 0.73 | 0.85 | 0.80 | 0.80 | 1.14 |
| **CLIPS** | gpt-6-luna (OpenAI Decisions API) | 0.79 | 0.85 | 0.80 | 0.80 | 1.08 |
| Language-only inverse planning | llama3.1 / judge / verbal / Jev / Decisions | 0.68 / 0.69 / 0.45 / 0.58 / 0.66 | 0.65–0.70 | 0.50–0.70 | 0.45–0.70 | 0.97–2.80 |
| Action-only inverse planning | – | 0.46 | 0.60 | 0.30 | 0.40 | 1.43 |
| Literal listener, naive | llama3.1 / yes/no judge | – | – | 0.46 / 0.63 | 0.61 / 0.57 | 1.40 / 1.19 |
| Literal listener, efficient | llama3.1 / yes/no judge | – | – | 0.47 / 0.73 | 0.64 / 0.69 | 1.40 / 1.12 |
| End to end (image + moves + sentence) | gpt-5.5 | 0.90\* | 0.90 | 0.75 | 0.85 | – |
| End to end (text map + moves + sentence) | Jev | 0.53 | 0.50 | 0.80 | 0.90 | – |
| End to end (text map + moves + sentence) | gpt-6-luna (Decisions) | 0.47 | 0.50 | 0.45 | 0.45 | – |

\* gpt-5.5 names one goal, so its P(g_true) is 0 or 1 per problem.

Thesis numbers (30 problems): CLIPS 0.74, precision / recall 0.97 / 0.97. Language-only 0.55, 0.90 / 0.83, length 1.26. Action-only 0.31, 0.43 / 0.40, length 1.68. Literal efficient 0.65 / 0.54. Literal naive 0.58 / 0.47. GPT-4V 0.29, 0.39 / 0.37.

**What reproduced.**
- The ordering CLIPS > language-only > action-only on goal accuracy.
- Literal listeners help less than CLIPS and make longer plans.
- Combining the two clues resolves ambiguity that neither resolves alone. With "Can you get me the red key?" and two red keys: language-only gives 0.50, action-only 0.38–0.60, CLIPS 0.71–0.83, and the robot fetches the correct red key.

**What didn't.**
- My language-only baseline is closer to CLIPS than in the thesis, because in about half my problems the sentence alone settles the goal.
- One problem (`tk_partial_door`) defeats every inverse-planning method. Two goals stay equally likely, and Q_MDP with a frozen posterior makes the robot pick up a key and then wait forever. This is a property of offline Q_MDP, not of the inference.
- gpt-5.5 is not the weakest method, as GPT-4V was in the thesis (see below).

10 problems is not the right sample size to judge anything, but I am happy with the work I was able to do.

## What I added beyond the reproduction

These came from trying to understand what the language model is really contributing, and from the practical problem that the thesis's LLMs no longer exist.

**1. The thesis's scoring method is not available anymore.** CLIPS needs log P(utterance | command): the probability that a speaker with that command in mind would say exactly this sentence. The thesis got it from OpenAI base models (curie, davinci-002) through the old Completions API, which returned log-probabilities for any text you sent. Those models are retired. Current chat models only report log-probabilities for tokens they generate themselves, so the sentence can't be scored directly.

**2. Teacher-forced scoring with a local model.** To keep the thesis's quantity, I score the sentence with llama3.1 on Ollama, one token at a time:
- I read the model's own BPE tokenizer out of its GGUF file, so every command is scored on the same token sequence.
- For each token, I ask for the top-20 next tokens after "prompt + sentence so far" and look up the target token's probability.

An earlier version that forced the whole sentence through constrained decoding was inconsistent: the token splits changed between calls, by up to 30 nats. Two limitations remain:
- Ollama shows only the top 20 tokens, so about 17% of tokens fall outside them and get an upper-bound score.
- llama3.1 is chat-tuned, while the thesis deliberately chose base models because chat models are over-confident.

**3. A yes/no judge instead of a speaker model.** With any chat API, I ask once per command: "Would a person plausibly say this utterance to give this command? Yes or No", and use log P("Yes") from the token log-probabilities. This flips the model from a speaker, P(u | c), to a listener, roughly P(c | u). With a uniform prior over the candidate commands that is proportional to the speaker likelihood, assuming the judge is calibrated (it isn't checked). It needs one call per command instead of one per token, and it did as well as the speaker model here (0.81 vs 0.79). I expected it to lose the speaker model's sensitivity to how informative the phrasing is (e.g. "a red key" vs "the red keys"), but 10 problems are too few to show that either way.

**4. Asking the model for a number does not work.** Same model, same question, but the model writes its own probability as Pydantic structured output: `{plausible, probability}`. Accuracy dropped from 0.81 to 0.64, and correct help from 0.90 to about 0.50. The written numbers are coarse (0.9, 0.8, 0.1) and similar across commands, so they separate goals too weakly for the robot to act. Top-1 stayed at 0.85: the ranking survives, but the robot is too unsure to act. Reading the log-probability of "Yes" is clearly better than asking for a number.

**5. Decision models as likelihood functions.** Models built to answer typed questions with calibrated probabilities fit CLIPS's scoring step naturally: one request per sentence with every command as a yes/no question, with no log-probability access needed. I tried two, [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (via OpenRouter) and the [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions) (gpt-6-luna). Both worked about as well as the log-probability scorers (0.73 and 0.79). They also fixed "Please unlock that door for me", the one problem where the llama scorer failed.

**6. The same models end to end.** Given the rules, the map, the moves and the sentence, and asked directly for the goal and what the robot should do:
- **Decision models were poor goal inferrers** (top-1 0.50). They follow the words and largely ignore the moves.
- **gpt-5.5 (in place of the retired GPT-4V) named the right goal in 9 of 10 problems.** The thesis's GPT-4V got 0.29. Its one miss was the problem where 50/50 is the correct answer: CLIPS reports the uncertainty, gpt-5.5 guesses. gpt-5.5 also got more than GPT-4V did: the moves as text and the walked path drawn on the image. And my problems are probably easier than the thesis's. So I can't separate "newer model" from "more information" from "easier problems".

**What I take from this.** LLMs are most useful inside CLIPS as narrow likelihood functions: "does this sentence fit this command?". Asked to do the whole inference, they did worse, apart from the largest reasoning model, which still gives no calibrated uncertainty and no plan. Inside CLIPS, the way the probability is extracted matters a lot. Token log-probabilities and calibrated decision models work, verbalised numbers don't.

## Differences from the thesis, and limitations

- **Problems:** 10 of my own instead of 30, and the gold assistance options come from my own optimal planner, not hand labels. With 10 problems, differences under about 0.1 are noise.
- **Not built:** VirtualHome, the human study (so the correlations with human ratings are not reproduced), and online belief updates during assistance.
- **Few-shot examples:** my own 14, since the thesis's 18 are only in its code.
- **Simulated human:** simplified. It also gives up when a joint state repeats, my addition, because a fixed-posterior Q_MDP can cycle.
- **Literal listener:** in its second phase the robot stays put but can still hand over a key it holds. Taken strictly, "only the principal moves" leaves a picked-up key stuck with the robot.
- **Caching:** LLM answers are cached (`results/cache/llm/`), so reruns reuse one sample from non-deterministic models (Jev, gpt-5.5).

## Running it

```bash
uv sync
uv run python -m pytest            # 62 tests: environments, planners, SIPS vs exact inference, CLIPS vs brute force, baselines
```

| Experiment | Command |
|---|---|
| Exp 1.1: SIPS vs BIRL, qualitative | `uv run python experiments/exp1_1_qualitative.py` |
| Exp 1.2: accuracy and cost (`dkg`, `taxi`, `blockwords`) | `uv run python experiments/exp1_2_accuracy.py dkg` |
| Exp 1.3: robustness | `uv run python experiments/exp1_3_robustness.py dkg` |
| Exp 2.1 / 2.2: planning mistakes | `uv run python experiments/exp2_1_dkg.py`, `experiments/exp2_2_blockwords.py` |
| Exp 3.2: open-ended goals | `uv run python experiments/exp3_2_engine.py` |
| **Exp 4.1: CLIPS and all baselines** | `uv run python experiments/exp4_1_mdkg.py` |

Exp 4.1 needs:
- Ollama running with `llama3.1` pulled.
- A `.env` file in this folder with `OPENAI_API_KEY=...` and `OPENROUTER_API_KEY=...` (for Jev).

`--no-openai` and `--no-jev` skip those models. The cached answers in `results/cache/llm/` let the experiment rerun without API calls. A first run from an empty cache takes over an hour, mostly the literal listener's Ollama calls.

## Code map

```
src/
  envs/         Doors-Keys-Gems, Taxi, Block Words (the single-agent domains)
  sips/         planners (A*, probabilistic A*), the agent model, SIPS, exact filtering
  birl/         BIRL baseline (exact, unbiased-async and oracle-async value iteration)
  mistakes/     Ch. 2: plan and action mistakes
  openended/    Ch. 3: open-ended goal inference
  clips/        Ch. 4: multi-agent DKG, joint planner, commands, inference, assistance,
                literal listener, rendering
  llms/         utterance scorers (Ollama, yes/no judge, verbal, decision models),
                the gpt-5.5 baseline, disk cache
  evaluation/   metrics and experiment harness
experiments/    one script per thesis experiment
checks/         correctness checks (SIPS → exact posterior, agent behaviour)
tests/
results/        tables (JSON), figures, LLM cache
```

## Reference

Tan Zhi-Xuan. *Scaling Cooperative Intelligence via Inverse Planning and Probabilistic Programming*. PhD thesis, MIT, 2025. CLIPS: T. Zhi-Xuan, L. Ying, V. Mansinghka, J. B. Tenenbaum, "Pragmatic Instruction Following and Goal Assistance via Cooperative Language-Guided Inverse Planning", AAMAS 2024. SIPS: T. Zhi-Xuan, J. Mann, T. Silver, J. B. Tenenbaum, V. Mansinghka, "Online Bayesian Goal Inference for Boundedly-Rational Planning Agents", NeurIPS 2020.
