"""The CLIPS utterance model, minus the LLM: plan -> salient actions -> command (thesis Sec. 4.2,
App. 4.8.1).

A command is a short ordered list of salient actions, lifted so that objects become typed
variables qualified by colour, and agents become indexicals:

    (handover you me ?key1) where (iscolor ?key1 red)

P(c | pi) is uniform over the valid subsets (size 1..K, in plan order) of the plan's salient
actions, with each lifted command collecting the mass of every subset that lifts to it.
"""
import itertools
from collections import defaultdict

from .env import MDKG

INDEXICAL = {"human": "me", "robot": "you"}

# Few-shot examples E for the utterance model. Our own; the thesis's 18 are only in its code.
EXAMPLES = [
    ("(handover you me ?key1) where (iscolor ?key1 blue)", "Hand me the blue key."),
    ("(unlock you ?door1) where (iscolor ?door1 red)", "Can you open the red door?"),
    ("(pickup you ?key1) where (iscolor ?key1 yellow)", "Could you pick up the yellow key?"),
    ("(handover you me ?key1) (handover you me ?key2) where (iscolor ?key1 green) (iscolor ?key2 red)",
     "Can you pass me the green and red keys?"),
    ("(pickup me ?key1) (unlock you ?door1) where (iscolor ?key1 blue) (iscolor ?door1 green)",
     "I'm getting the blue key, can you open the green door?"),
    ("(unlock you ?door1) (unlock you ?door2) where (iscolor ?door1 blue) (iscolor ?door2 blue)",
     "Please unlock the blue doors."),
    ("(pickup you ?key1) (pickup you ?key2) where (iscolor ?key1 red) (iscolor ?key2 red)",
     "Get the two red keys."),
    ("(handover you me ?key1) where (iscolor ?key1 red)", "I need a red key."),
    ("(unlock you ?door1) where (iscolor ?door1 yellow)", "Unlock the yellow door for me."),
    ("(pickup me ?key1) (handover you me ?key2) where (iscolor ?key1 red) (iscolor ?key2 blue)",
     "I'll grab the red key, you bring me the blue one."),
    ("(pickup you ?key1) where (iscolor ?key1 blue)", "Go get the blue key."),
    ("(unlock me ?door1) (unlock you ?door2) where (iscolor ?door1 red) (iscolor ?door2 blue)",
     "I'll open the red door, you open the blue one."),
    ("(pickup you ?key1) (unlock you ?door1) where (iscolor ?key1 red) (iscolor ?door1 green)",
     "Pick up the red key and unlock the green door."),
    ("(handover you me ?key1) (unlock you ?door1) where (iscolor ?key1 yellow) (iscolor ?door1 blue)",
     "Give me the yellow key and open the blue door."),
]


def salient_actions(env: MDKG, rollout) -> list[tuple[str, str, str]]:
    """Salient actions (verb, actor, object) along a rollout of (state, action) pairs."""
    return [sa for s, a in rollout if (sa := env.salient(s, a)) is not None]


def valid(subset) -> bool:
    """Pruning rules of thesis App. 4.8.1 (mDKG): at most two distinct action types, at least one
    action by the listener, and no chains of dependent actions (two actions on the same object,
    e.g. picking up a key and then handing over that same key)."""
    if len({verb for verb, _, _ in subset}) > 2:
        return False
    if all(actor == "human" for _, actor, _ in subset):
        return False
    objs = [obj for _, _, obj in subset]
    return len(set(objs)) == len(objs)


def lift(env: MDKG, subset) -> str:
    """Replace objects by typed variables (numbered by first appearance) and agents by me/you."""
    names: dict[str, str] = {}
    counts: dict[str, int] = defaultdict(int)
    parts, preds = [], []
    for verb, actor, obj in subset:
        if obj not in names:
            kind = "key" if obj.startswith("key") else "door"
            counts[kind] += 1
            names[obj] = f"?{kind}{counts[kind]}"
            preds.append(f"(iscolor {names[obj]} {env.color_of(obj)})")
        who = INDEXICAL[actor]
        if verb == "handover":
            parts.append(f"(handover {who} {INDEXICAL['human' if actor == 'robot' else 'robot']} {names[obj]})")
        else:
            parts.append(f"({verb} {who} {names[obj]})")
    return " ".join(parts) + (" where " + " ".join(preds) if preds else "")


def command_distribution(env: MDKG, salient, K: int = 3) -> dict[str, float]:
    """P(c | pi): uniform over the valid ordered subsets of size 1..K of the salient actions,
    summed over subsets that lift to the same command. Empty if nothing is communicable."""
    mass: dict[str, float] = defaultdict(float)
    subsets = [sub for k in range(1, K + 1) for sub in itertools.combinations(salient, k) if valid(sub)]
    for sub in subsets:
        mass[lift(env, sub)] += 1.0 / len(subsets)
    return dict(mass)
