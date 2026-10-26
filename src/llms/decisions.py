"""Decision models: answer typed questions with probabilities instead of writing text.

Two of them, behind one small interface:
  Jev (TypeSafe, via OpenRouter's alpha decisions endpoint)    noul / choice questions
  OpenAI Decisions API (gpt-6-luna)                            predicate / choice questions
A yes/no question comes back as one probability; a choice question as a probability per option.

They are used two ways:
  DecisionScorer      a CLIPS utterance scorer: one yes/no question per command ("would a person
                      plausibly say this utterance to give this command?"), all in one request;
                      score = log(max(P(yes), FLOOR)). A calibrated stand-in for log P(u | c).
  DecisionBaseline    end to end, like the multimodal baseline: the rules, the map as text, the
                      moves and the instruction in; P(goal) for each gem and P(should help) for
                      each key or door out.
Both models report probabilities to two decimals, so P(yes) is floored at FLOOR before the log.
"""
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

from .base import ScoreCache, Scorer, load_env
from .openai_judge import SYSTEM

FLOOR = 1e-3
QUESTION = 'Command: {c}\nWould a person plausibly say the utterance "{u}" to give this command?'


class Jev:
    url = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, model: str = "~typesafe/jev-latest"):
        load_env()
        self.model, self.name = model, "jev"

    def ask(self, state, yesno: dict[str, str], choices: dict[str, tuple[str, dict[str, str]]] = {}) -> dict:
        """yesno: name -> instructions. choices: name -> (instructions, {option: description}).
        Returns name -> P(yes), or name -> {option: probability}."""
        questions = {n: {"type": "noul", "instructions": t} for n, t in yesno.items()}
        questions |= {n: {"type": "choice", "instructions": t, "criteria": opts} for n, (t, opts) in choices.items()}
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json",
            "X-OpenRouter-Title": "clips-reproduction"})
        answers = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    answers = json.loads(r.read())["answers"]
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 529) or attempt == 4:
                    raise RuntimeError(f"Jev HTTP {e.code}: {e.read().decode()[:300]}") from None
                time.sleep(2 ** attempt)
        assert answers is not None
        out = {n: float(answers[n]["noul"]) for n in yesno}
        out |= {n: {k: float(v) for k, v in answers[n]["probabilities"].items()} for n in choices}
        return out


class OpenAIDecisions:
    def __init__(self, model: str = "gpt-6-luna"):
        load_env()
        from openai import OpenAI

        self.client, self.model, self.name = OpenAI(), model, "decisions-" + model

    def ask(self, state, yesno: dict[str, str], choices: dict[str, tuple[str, dict[str, str]]] = {}) -> dict:
        questions = [{"type": "predicate", "name": n, "instructions": t} for n, t in yesno.items()]
        questions += [{"type": "choice", "name": n, "instructions": t,
                       "choices": [{"value": k, "description": v} for k, v in opts.items()]}
                      for n, (t, opts) in choices.items()]
        state_text = state if isinstance(state, str) else json.dumps(state)
        d = self.client.decisions.create(model=self.model, input=state_text, questions=questions)
        answers = {a.name: a for a in d.answers}
        refused = [n for n, a in answers.items() if a.type == "refusal"]
        assert not refused, f"refused: {refused}"
        out = {n: float(answers[n].probability) for n in yesno}
        out |= {n: {p.value: float(p.probability) for p in answers[n].probabilities} for n in choices}
        return out


class DecisionScorer(Scorer):
    """log P(yes) per command from a decision model, all commands in one request (chunks of 200)."""

    def __init__(self, model):
        self.model, self.name = model, model.name + "-scorer"
        self.cache = ScoreCache(self.name)

    def loglik(self, utterance, commands, examples) -> np.ndarray:
        shots = "".join(f'Command: {c}\nUtterance: "{u}"\n\n' for c, u in examples)
        state = SYSTEM + shots + f'The human player just said: "{utterance}"'
        keys = [ScoreCache.key("decision-v1", self.model.name, shots, c, utterance) for c in commands]
        todo = [(k, c) for k, c in zip(keys, commands) if self.cache.get(k) is None]
        for i in range(0, len(todo), 200):
            chunk = todo[i:i + 200]
            ans = self.model.ask(state, {f"c{j}": QUESTION.format(c=c, u=utterance) for j, (_, c) in enumerate(chunk)})
            for j, (k, _) in enumerate(chunk):
                self.cache.put(k, ans[f"c{j}"])
            self.cache.save()
        return np.log(np.maximum([self.cache.get(k) for k in keys], FLOOR))


E2E_RULES = """A human and a robot play a grid game together. The human wants to collect exactly one of the four gems; the robot helps.
Rules: the robot cannot pick up gems. Players move on floor squares. Keys unlock doors of the same colour, and each key works once.
One player can pass a key to the other when they are next to each other or on the same square."""


def e2e_state(description: str, moves: list[str], utterance: str) -> str:
    did = (f"The human took these actions: {', '.join(moves)} (the robot waited), then said" if moves
           else "The human has not moved yet, and said")
    return f"{E2E_RULES}\n\n{description}\n\n{did}: \"{utterance}\""


class DecisionBaseline:
    """End to end: P(goal) and P(the robot should do this) for each key or door."""

    def __init__(self, model):
        self.model, self.name = model, "decision-e2e-" + model.name
        self.cache = ScoreCache(self.name)

    def ask(self, state_text: str, goals: list[str], candidates: dict[str, str], options: str) -> dict:
        verb = "pick up" if options == "keys" else "unlock"
        yesno = {o: f"Should the robot {verb} {o} ({desc}) to best assist the human?" for o, desc in candidates.items()}
        choices = {"goal": ("Which gem is most likely the human's goal?", {g: f"the {g} gem" for g in goals})}
        key = ScoreCache.key("e2e-v1", self.model.name, state_text, json.dumps(yesno), json.dumps(choices))
        hit = self.cache.get(key)
        if hit is None:
            hit = self.model.ask(state_text, yesno, choices)
            self.cache.put(key, hit)
            self.cache.save()
        return hit
