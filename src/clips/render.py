"""Draw an mDKG state as a PNG for the multimodal LLM baseline.

Like the thesis's stimulus frames: white floor, dark walls, keys and doors labelled in the top-right
corner of their cell. We also draw the human's path so far as a dashed line, because the model
sees one still image rather than the animation the human raters saw.
"""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

from .env import HELD_H, HELD_R, ON_MAP, MDKG

COLORS = {"r": "#d6463a", "y": "#e0a81c", "b": "#2e6bd4", "g": "#2b9659"}
GEM_COLORS = {"red": "#d6463a", "yellow": "#e0a81c", "blue": "#2e6bd4", "green": "#2b9659"}


def render_png(env: MDKG, s, trail=(), cell: float = 0.9) -> bytes:
    _, hr, hc, rr, rc, keyloc, locked, gem = s
    R, C = len(env.rows), len(env.rows[0])
    fig, ax = plt.subplots(figsize=(C * cell, R * cell + 0.5), dpi=110)
    ax.set_xlim(0, C)
    ax.set_ylim(R, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for r, row in enumerate(env.rows):
        for c, ch in enumerate(row):
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#4a4a4a" if ch == "#" else "white", edgecolor="#bbbbbb", lw=0.6))
    for d, (r, c, col) in enumerate(env.doors):
        if (locked >> d) & 1:
            ax.add_patch(Rectangle((c + 0.06, r + 0.06), 0.88, 0.88, facecolor=COLORS[col], edgecolor="black", lw=1))
            ax.add_patch(Circle((c + 0.5, r + 0.45), 0.09, facecolor="black"))
            ax.add_patch(Rectangle((c + 0.46, r + 0.48), 0.08, 0.2, facecolor="black"))
            ax.text(c + 0.95, r + 0.04, env.door_names[d], ha="right", va="top", fontsize=6.5, color="white", weight="bold")
    for g, (r, c) in enumerate(env.gems):
        if gem != g:
            x, y = c + 0.5, r + 0.5
            ax.add_patch(Polygon([(x, y - 0.32), (x + 0.26, y - 0.08), (x, y + 0.32), (x - 0.26, y - 0.08)],
                                 facecolor=GEM_COLORS[env.goal_names[g]], edgecolor="black", lw=0.8))
    for k, (r, c, col) in enumerate(env.keys):
        if keyloc[k] == ON_MAP:
            _key(ax, c + 0.45, r + 0.55, COLORS[col])
            ax.text(c + 0.95, r + 0.04, env.key_names[k], ha="right", va="top", fontsize=6.5, color="black", weight="bold")
    if len(trail) > 1:
        ax.plot([c + 0.5 for _, c in trail], [r + 0.5 for r, _ in trail], ls="--", lw=2, color="#333333", zorder=3)
    ax.add_patch(Rectangle((rc + 0.2, rr + 0.2), 0.6, 0.6, facecolor="#5b4fd6", edgecolor="black", zorder=4))
    ax.text(rc + 0.5, rr + 0.5, "R", ha="center", va="center", color="white", fontsize=10, weight="bold", zorder=5)
    ax.add_patch(Circle((hc + 0.5, hr + 0.5), 0.3, facecolor="#222222", edgecolor="black", zorder=4))
    ax.text(hc + 0.5, hr + 0.5, "H", ha="center", va="center", color="white", fontsize=10, weight="bold", zorder=5)
    held = {who: [env.key_names[k] for k, loc in enumerate(keyloc) if loc == code] for who, code in (("human", HELD_H), ("robot", HELD_R))}
    ax.text(0, R + 0.35, f"Human holds: {', '.join(held['human']) or 'nothing'}    Robot holds: {', '.join(held['robot']) or 'nothing'}",
            fontsize=8, va="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _key(ax, x, y, color):
    ax.add_patch(Circle((x - 0.15, y), 0.11, facecolor="none", edgecolor=color, lw=2.5))
    ax.add_patch(Rectangle((x - 0.05, y - 0.035), 0.32, 0.07, facecolor=color))
    ax.add_patch(Rectangle((x + 0.17, y), 0.05, 0.1, facecolor=color))


def describe_text(env: MDKG, s) -> str:
    """The state as text: an ASCII map with a legend, then every object with its position."""
    _, hr, hc, rr, rc, keyloc, locked, gem = s
    grid = env.render_text(s).splitlines()[:len(env.rows)]
    lines = ["Map (row 0 is the top, column 0 is the left):", *grid,
             "Legend: # wall, . floor, @ human, & robot, digits are gems (1 red, 2 yellow, 3 blue, 4 green), "
             "capital letters are locked doors and small letters are keys (r red, y yellow, b blue, g green).",
             f"The human is at row {hr}, column {hc}. The robot is at row {rr}, column {rc}."]
    for g, (r, c) in enumerate(env.gems):
        if gem != g:
            lines.append(f"The {env.goal_names[g]} gem is at row {r}, column {c}.")
    for k, desc in key_descriptions(env, s).items():
        lines.append(f"{k}: {desc}.")
    for d, desc in door_descriptions(env, s).items():
        lines.append(f"{d}: {desc}.")
    held = {who: [env.key_names[k] for k, loc in enumerate(keyloc) if loc == code] for who, code in (("human", HELD_H), ("robot", HELD_R))}
    lines.append(f"Human holds: {', '.join(held['human']) or 'nothing'}. Robot holds: {', '.join(held['robot']) or 'nothing'}.")
    return "\n".join(lines)


NAMES = {"r": "red", "y": "yellow", "b": "blue", "g": "green"}


def key_descriptions(env: MDKG, s) -> dict[str, str]:
    """Keys still on the map: name -> 'red key at row 5, column 1'."""
    return {env.key_names[k]: f"{NAMES[col]} key at row {r}, column {c}"
            for k, (r, c, col) in enumerate(env.keys) if s[5][k] == ON_MAP}


def door_descriptions(env: MDKG, s) -> dict[str, str]:
    """Locked doors: name -> 'red door at row 2, column 1'."""
    return {env.door_names[d]: f"{NAMES[col]} door at row {r}, column {c}"
            for d, (r, c, col) in enumerate(env.doors) if (s[6] >> d) & 1}
