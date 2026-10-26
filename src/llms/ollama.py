"""Utterance likelihoods log P(u | c) from a local Ollama model, by teacher forcing.

Ollama cannot score a given text in one call: it has no echo and no tokenize endpoint, and it
reports log-probabilities only for tokens it generates, at most the top 20 per position. So:

1. Tokenise the utterance ourselves with the model's own byte-level BPE, read from its GGUF file.
   Every command is then scored on the same token sequence.
2. For each token, ask for one generated token after "prompt + utterance so far" and read the
   target token's log-probability from the top 20.
3. If the target is not in the top 20, its log-probability is at most the 20th one; we use that
   upper bound and count the token as censored. Censoring caps how strongly a very unlikely word
   counts against a command, so it compresses contrasts; the experiment reports how often it
   happens.

(A first version forced the whole utterance with a JSON-schema constraint. Its token splits
varied between calls and commands, e.g. '"C' + 'an' vs '"' + 'Can', and each odd split cost
10-17 nats, more than the differences between commands. Do not go back to it.)
"""
import json
import urllib.request
from functools import lru_cache

import numpy as np
import regex

from .base import ScoreCache, Scorer

HEADER = "Translate each command into something a person playing the game might say.\n\n"

# Pre-tokeniser of the llama-bpe (Llama 3) tokenizer.
LLAMA3_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")


def build_prompt(examples, command: str) -> str:
    """Few-shot prompt ending just after the opening quote of the utterance."""
    shots = "".join(f'Command: {c}\nUtterance: "{u}"\n\n' for c, u in examples)
    return HEADER + shots + f'Command: {command}\nUtterance: "'


def _bytes_to_unicode() -> dict:
    """GPT-2's reversible map from bytes to printable characters (byte-level BPE vocabularies)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class BPETokenizer:
    """Byte-level BPE from a merges list ("a b" strings, best first)."""

    def __init__(self, merges: list[str], split_pattern: str = LLAMA3_SPLIT):
        self.ranks = {tuple(m.split(" ")): i for i, m in enumerate(merges)}
        self.split = regex.compile(split_pattern)
        self.byte_map = _bytes_to_unicode()
        self.unmap = {v: k for k, v in self.byte_map.items()}

    @lru_cache(maxsize=4096)
    def _bpe(self, word: str) -> tuple:
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get((a, b), np.inf), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))]
            rank, i = min(pairs)
            if rank == np.inf:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return tuple(parts)

    def encode(self, text: str) -> list[str]:
        """Tokens as decoded text pieces."""
        out = []
        for piece in self.split.findall(text):
            mapped = "".join(self.byte_map[b] for b in piece.encode())
            out.extend(bytes(self.unmap[ch] for ch in tok).decode() for tok in self._bpe(mapped))
        return out

    @classmethod
    def from_gguf(cls, path: str) -> "BPETokenizer":
        from gguf import GGUFReader

        f = GGUFReader(path).fields
        assert bytes(f["tokenizer.ggml.model"].parts[-1]).decode() == "gpt2", "only byte-level BPE models"
        assert bytes(f["tokenizer.ggml.pre"].parts[-1]).decode() == "llama-bpe", "only the Llama 3 pre-tokeniser"
        m = f["tokenizer.ggml.merges"]
        return cls([bytes(m.parts[i]).decode() for i in m.data])


class OllamaScorer(Scorer):
    def __init__(self, model: str = "llama3.1:latest", host: str = "http://localhost:11434"):
        self.model, self.host = model, host
        self.name = "ollama-" + model.replace(":", "_").replace("/", "_")
        self.cache = ScoreCache(self.name)
        self._tok = None
        self.n_tokens = self.n_censored = 0  # over every score computed (not cached) in this run

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())

    @property
    def tokenizer(self) -> BPETokenizer:
        if self._tok is None:
            modelfile = self._post("/api/show", {"model": self.model})["modelfile"]
            path = next(line.split(" ", 1)[1] for line in modelfile.splitlines() if line.startswith("FROM /"))
            self._tok = BPETokenizer.from_gguf(path)
        return self._tok

    def _top20(self, prompt: str) -> dict:
        out = self._post("/api/generate", {
            "model": self.model, "prompt": prompt, "raw": True, "stream": False,
            "options": {"temperature": 0, "num_predict": 1, "seed": 0},
            "logprobs": True, "top_logprobs": 20,
        })
        return {t["token"]: t["logprob"] for t in out["logprobs"][0]["top_logprobs"]}

    def score_one(self, utterance: str, command: str, examples) -> float:
        """log P(utterance, closing quote | prompt), one forced token at a time."""
        prompt = build_prompt(examples, command)
        tokens = self.tokenizer.encode(utterance) + ['"']
        total, prefix = 0.0, ""
        for tok in tokens:
            top = self._top20(prompt + prefix)
            self.n_tokens += 1
            if tok in top:
                total += top[tok]
            else:
                total += min(top.values())
                self.n_censored += 1
            prefix += tok
        return total

    def loglik(self, utterance, commands, examples) -> np.ndarray:
        shots = json.dumps(examples)
        scores = []
        for c in commands:
            key = ScoreCache.key("teacher-forced-v2", self.model, shots, c, utterance)
            v = self.cache.get(key)
            if v is None:
                v = self.score_one(utterance, c, examples)
                self.cache.put(key, v)
            scores.append(v)
        self.cache.save()
        return np.array(scores)
