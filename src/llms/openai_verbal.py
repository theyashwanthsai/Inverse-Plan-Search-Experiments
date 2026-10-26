"""A third utterance scorer: an OpenAI chat model writes its own probability, as structured output.

Same question as the yes/no judge (openai_judge.py), but instead of reading the log-probability of
"Yes", the answer is parsed into a Pydantic object {plausible, probability} and log(probability) is
the score. This tests whether a verbalised number can stand in for token log-probabilities.

Verbalised probabilities are coarse (models favour round numbers such as 0.9 or 0.1) and can be 0,
which would rule a goal out on one word of the model's. They are floored at FLOOR.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import BaseModel, Field

from .base import ScoreCache, Scorer, load_env
from .openai_judge import SYSTEM

FLOOR = 1e-3
QUESTION = ('Command: {c}\nUtterance: "{u}"\n\nWould a person plausibly say this utterance to give this command? '
            "Also give the probability, between 0 and 1, that a person giving this command would say this utterance.")


class Judgement(BaseModel):
    plausible: bool
    probability: float = Field(ge=0.0, le=1.0)


class OpenAIVerbalScorer(Scorer):
    def __init__(self, model: str = "gpt-4o-mini", workers: int = 8):
        load_env()
        from openai import OpenAI

        self.client, self.model, self.workers = OpenAI(), model, workers
        self.name = "openai-verbal-" + model
        self.cache = ScoreCache(self.name)

    def judge(self, utterance: str, command: str, examples) -> Judgement:
        shots = "".join(f'Command: {c}\nUtterance: "{u}"\n\n' for c, u in examples)
        r = self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM + shots},
                      {"role": "user", "content": QUESTION.format(c=command, u=utterance)}],
            response_format=Judgement, temperature=0, seed=0,
        )
        out = r.choices[0].message.parsed
        assert out is not None, "model refused or returned no parsable answer"
        return out

    def loglik(self, utterance, commands, examples) -> np.ndarray:
        shots = json.dumps(examples)
        keys = [ScoreCache.key("verbal-v1", self.model, shots, c, utterance) for c in commands]
        todo = [(k, c) for k, c in zip(keys, commands) if self.cache.get(k) is None]
        with ThreadPoolExecutor(self.workers) as pool:
            for (k, _), j in zip(todo, pool.map(lambda kc: self.judge(utterance, kc[1], examples), todo)):
                self.cache.put(k, j.model_dump())
        self.cache.save()
        return np.array([np.log(max(self.cache.get(k)["probability"], FLOOR)) for k in keys])
