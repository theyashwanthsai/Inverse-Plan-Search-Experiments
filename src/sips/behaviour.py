"""Descriptive statistics of trajectories, used to check that the agent model is not just optimal."""
from .planners import astar


def revisits_cell(states) -> bool:
    """True if the agent returns to a grid cell it had left (a backtrack)."""
    cells = [s[:2] for s in states]
    seen = set()
    prev = None
    for c in cells:
        if c != prev:
            if c in seen:
                return True
            seen.add(c)
        prev = c
    return False


def summarize(domain, trajectories_by_goal: dict, admissible_h) -> dict:
    """Per goal: fraction reaching the goal, fraction optimal, fraction backtracking,
    fraction that made the goal unreachable at some point."""
    out = {}
    for g, trajs in trajectories_by_goal.items():
        opt_len = len(astar(domain, domain.initial_state, g, admissible_h)[0])
        alive = domain.states_that_can_reach(g)
        n = len(trajs)
        reached = [domain.is_goal(t[-1], g) for t in trajs]
        out[domain.goal_names[g]] = {
            "n": n,
            "reached": sum(reached) / n,
            "optimal": sum(r and len(t) - 1 == opt_len for r, t in zip(reached, trajs)) / n,
            "backtracked": sum(revisits_cell(t) for t in trajs) / n,
            "goal_made_unreachable": sum(any(s not in alive for s in t) for t in trajs) / n,
            "mean_len": sum(len(t) - 1 for t in trajs) / n,
            "optimal_len": opt_len,
        }
    return out
