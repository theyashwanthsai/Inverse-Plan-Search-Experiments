"""A cheap stand-in for log P(u | c) from an OpenAI chat model.

OpenAI no longer serves models that can score a given text (davinci-002, babbage-002 and
gpt-3.5-turbo-instruct all return "deprecated"), and chat models only report log-probabilities of
tokens they generate. So instead we ask, per command, whether a person would plausibly say the
utterance to give that command, and use log P("Yes") as the score.

Why that is a likelihood up to a constant (an approximation, stated in the write-up): if the judge
were a calibrated listener over the candidate commands, P(c | u) = P(Yes | u, c) / Z(u), and with a
uniform prior over candidates P(u | c) is proportional to P(c | u). Neither condition is checked.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .base import ScoreCache, Scorer, load_env

SYSTEM = (
    "In a grid game, a human and a robot cooperate. The human gives the robot short instructions. "
    "Instructions summarise a formal command. In commands, 'you' is the robot, 'me' is the human, "
    "?key1/?door1 are objects, and (iscolor ?x c) gives an object's colour. Examples:\n\n"
)
QUESTION = 'Command: {c}\nUtterance: "{u}"\n\nWould a person plausibly say this utterance to give this command? Answer Yes or No.'


class OpenAIJudgeScorer(Scorer):
    def __init__(self, model: str = "gpt-4o-mini", workers: int = 8):
        load_env()
        from openai import OpenAI

        self.client, self.model, self.workers = OpenAI(), model, workers
        self.name = "openai-judge-" + model
        self.cache = ScoreCache(self.name)

    def score_one(self, utterance: str, command: str, examples) -> float:
        shots = "".join(f'Command: {c}\nUtterance: "{u}"\n\n' for c, u in examples)
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM + shots},
                      {"role": "user", "content": QUESTION.format(c=command, u=utterance)}],
            max_tokens=1, temperature=0, logprobs=True, top_logprobs=10, seed=0,
        )
        lp = r.choices[0].logprobs
        assert lp is not None and lp.content, "model returned no logprobs"
        top = lp.content[0].top_logprobs
        yes = [t.logprob for t in top if t.token.strip().lower() == "yes"]
        if yes:
            return float(np.logaddexp.reduce(yes))
        # "Yes" fell outside the top 10: its probability is at most that of the 10th token.
        return float(min(t.logprob for t in top))

    def loglik(self, utterance, commands, examples) -> np.ndarray:
        shots = json.dumps(examples)
        keys = [ScoreCache.key(self.model, shots, c, utterance) for c in commands]
        todo = [(k, c) for k, c in zip(keys, commands) if self.cache.get(k) is None]
        with ThreadPoolExecutor(self.workers) as pool:
            for (k, _), v in zip(todo, pool.map(lambda kc: self.score_one(utterance, kc[1], examples), todo)):
                self.cache.put(k, v)
        self.cache.save()
        return np.array([self.cache.get(k) for k in keys])
