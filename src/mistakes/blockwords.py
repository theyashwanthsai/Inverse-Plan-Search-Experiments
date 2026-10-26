"""Block Words for the mistake model: candidate words plus the mis-spelled targets an agent can
briefly pursue ("paer" for "pear")."""
from envs.blockwords import BlockWords
from envs.blockwords.env import INITIAL_CONFIGS
from .agent import MistakeParams
from sips.agent import AgentParams

WORDS = ["pear", "reap", "pace", "wear", "draw"]  # my choice: the thesis's five words are not given


def corruptions(word: str, candidates=WORDS) -> list[str]:
    """Adjacent-letter swaps that are not themselves candidate words."""
    out = []
    for i in range(len(word) - 1):
        w = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if w != word and w not in candidates and w not in out:
            out.append(w)
    return out


def setup(config: int = 0, words=WORDS, eps_g=0.1, recover=0.3, epsilon=0.05):
    """(domain, MistakeParams): goals 0..len(words)-1 are the candidates; later ones are corruptions."""
    extra, similar = [], []
    for w in words:
        idx = []
        for c in corruptions(w, words):
            if c not in extra:
                extra.append(c)
            idx.append(len(words) + extra.index(c))
        similar.append(tuple(idx))
    domain = BlockWords(INITIAL_CONFIGS[config], words=list(words) + extra, name=f"blockwords-m{config}")
    mp = MistakeParams(base=AgentParams(epsilon=epsilon, heuristic="hadd"), eps_g=eps_g, recover=recover,
                       similar=tuple(similar), n_obs=len(words))
    return domain, mp
