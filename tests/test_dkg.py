from collections import deque

from sips.envs.dkg_maps import backtrack_map, failure_map, tiny_map


def bfs_plan(env, s0, g):
    """Shortest action sequence from s0 to goal g, or None if unreachable (exhaustive search)."""
    parent = {s0: None}
    queue = deque([s0])
    while queue:
        s = queue.popleft()
        if env.is_goal(s, g):
            plan = []
            while parent[s] is not None:
                s, a = parent[s]
                plan.append(a)
            return plan[::-1]
        for a, t in env.successors(s):
            if t not in parent:
                parent[t] = (s, a)
                queue.append(t)
    return None


def run(env, actions, s=None):
    s = env.initial_state if s is None else s
    for a in actions:
        assert a in env.legal_actions(s), (a, env.render_text(s))
        s = env.step(s, a)
    return s


def test_locked_door_blocks_and_key_is_single_use():
    env = tiny_map()
    s = env.initial_state
    assert "down" not in env.legal_actions(s)  # door below the start is locked
    assert "unlock_down" not in env.legal_actions(s)  # no key yet
    s = run(env, ["left", "pickup", "right"], s)
    assert s[2] == (1,)
    s = run(env, ["unlock_down"], s)
    assert s[2] == (0,), "unlocking consumes the key"
    assert "down" in env.legal_actions(s)


def test_pickup_gem_achieves_goal():
    env = tiny_map()
    s = run(env, ["left", "left", "pickup"])
    assert env.is_goal(s, 0) and not env.is_goal(s, 1)


def test_all_gems_reachable_from_start():
    for env in [backtrack_map(), failure_map(), tiny_map()]:
        for g in range(env.n_goals):
            assert bfs_plan(env, env.initial_state, g) is not None, (env.name, env.goal_names[g])


def test_failure_map_wasting_key_locks_blue_away():
    env = failure_map()
    blue = env.goal_names.index("blue")
    # Optimal solution exists from the start, and it spends the first key on the bottom door.
    plan = bfs_plan(env, env.initial_state, blue)
    assert plan is not None
    # The myopic mistake: grab the near key and use it on the first door towards blue.
    s = run(env, ["right", "right", "pickup", "right", "unlock_right"])
    assert bfs_plan(env, s, blue) is None, "blue should be unreachable after wasting the key"
    assert not any(env.is_goal(t, blue) for t in env.reachable_states(s))


def test_backtrack_map_optimal_blue_goes_left_first():
    env = backtrack_map()
    blue = env.goal_names.index("blue")
    plan = bfs_plan(env, env.initial_state, blue)
    assert plan[0] == "left", "optimal blue plan fetches the far-left key first"
    # The myopic route (near key, first door) still leaves blue reachable, but only by backtracking.
    myopic = ["right", "right", "pickup", "right", "unlock_right", "right", "right", "right", "right"]
    s = run(env, myopic)
    rest = bfs_plan(env, s, blue)
    assert rest is not None and rest[0] == "left" and len(plan) < len(myopic) + len(rest)
