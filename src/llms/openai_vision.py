"""Multimodal LLM baseline (thesis Sec. 4.5.2, "GPT-4V"): give the model everything and ask.

The thesis prompted gpt-4-1106-vision-preview zero-shot with the rules shown to its human raters,
the final frame of the stimulus, and the instruction, and asked for the goal and the keys to pick
up as JSON. GPT-4V is retired; we use a current GPT-5 model with the same prompt, two additions
(a legend for the image, and the human's moves as text, since a still frame does not show them),
and a Pydantic schema in place of "respond in the following JSON format".
"""
import base64
import hashlib
from typing import Literal

from pydantic import BaseModel

from .base import ScoreCache, load_env

RULES = """You're watching a human and a robot collaborating on a treasure game shown below. The human player wants to collect exactly one of the four target gems. The two players cooperate to achieve the human's goal.
During the game, the human may communicate with the robot to coordinate their actions. Any communication that happens is shown below the visuals.
The rules of the game are as follows:
- The human player's goal is to collect one target gem.
- The robot plays an assistive role, and cannot pick up gems on its own.
- The players can move on the white squares.
- The players have a full view of the map at all times.
- Keys are used to unlock doors with the same color (e.g. red keys only unlock red doors).
- Each key can only be used once. Keys disappear after being used.
- One player can pass keys to the other if they're next to each other.
- Players can occupy the same square and walk past each other.
- The keys and doors are labeled. The labels are shown on the top right corner of each cell.
In the image, the human is the black circle marked H and the robot is the purple square marked R. The dashed line is the path the human walked."""

QUESTION = {
    "keys": "Which key(s) should the robot pick up to best assist the human?",
    "doors": "Which door(s) should the robot unlock to best assist the human?",
}


class Answer(BaseModel):
    goal: Literal["red", "yellow", "blue", "green"]
    assist: list[str]


def build_prompt(moves: list[str], utterance: str, options: str) -> str:
    did = (f"the human player took these actions: {', '.join(moves)} (the robot waited), then gave the instruction"
           if moves else "the human player has not moved yet, and gave the instruction")
    return (f"{RULES}\n\nIn the scenario above, {did}: \"{utterance}\"\n"
            f"Given this information, what gem is most likely to be the human agent's goal? {QUESTION[options]}\n"
            "Answer with the goal colour and the list of labels (e.g. key1, door2).")


class VisionBaseline:
    def __init__(self, model: str = "gpt-5.5"):
        load_env()
        from openai import OpenAI

        self.client, self.model = OpenAI(), model
        self.name = "vision-" + model
        self.cache = ScoreCache(self.name)

    def ask(self, png: bytes, moves: list[str], utterance: str, options: str) -> Answer:
        prompt = build_prompt(moves, utterance, options)
        key = ScoreCache.key("vision-v1", self.model, prompt, hashlib.sha256(png).hexdigest())
        hit = self.cache.get(key)
        if hit is None:
            image = "data:image/png;base64," + base64.b64encode(png).decode()
            r = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}},
                                                       {"type": "text", "text": prompt}]}],
                response_format=Answer,
            )
            parsed = r.choices[0].message.parsed
            assert parsed is not None, "model refused or returned no parsable answer"
            hit = parsed.model_dump()
            self.cache.put(key, hit)
            self.cache.save()
        return Answer(**hit)
