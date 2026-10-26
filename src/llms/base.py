"""Shared pieces for LLM utterance scorers.

A scorer turns (utterance, candidate commands, few-shot examples) into one log-score per command.
CLIPS only uses the scores up to an additive constant shared by all commands (the utterance is
fixed), so a scorer may return log P(u | c) or anything proportional to it in c.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from common.paths import RESULTS

CACHE_DIR = RESULTS / "cache" / "llm"


class ScoreCache:
    """A JSON file of key -> float, so reruns never repeat an LLM call."""

    def __init__(self, name: str):
        self.path = CACHE_DIR / f"{name}.json"
        self.data: dict = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value: float) -> None:
        self.data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        tmp.replace(self.path)


class Scorer:
    name: str

    def loglik(self, utterance: str, commands: list[str], examples: list[tuple[str, str]]) -> np.ndarray:
        raise NotImplementedError


def load_env() -> None:
    """Load API keys from sips/.env, overriding anything already in the environment."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
