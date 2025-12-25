from typing import Dict
from env.gridworld import GridWorld


def expert_action(env: GridWorld, obs: Dict) -> int:
    """
    Oracle policy with full state access.
    Given the instruction "pick up the X block", it:
    - finds the target object
    - moves toward it
    - picks it up when on the same cell
    """

    instr = obs["instruction"].lower()
    agent_r, agent_c = obs["agent_pos"]

    # Parse target color
    target_color = None
    if "pick up the" in instr:
        parts = instr.split()
        try:
            idx = parts.index("the")
            target_color = parts[idx + 1]
        except Exception:
            pass

    if target_color is None:
        return GridWorld.UP  # fallback noop-ish

    # If already holding target, do nothing
    if obs["holding"] == target_color:
        return GridWorld.DROP  # or no-op placeholder

    # Find target object position
    target_pos = None
    for o in obs["objects"]:
        if o["color"] == target_color:
            target_pos = o["pos"]
            break

    if target_pos is None:
        return GridWorld.UP

    tr, tc = target_pos

    # If on the target, pick it up
    if (agent_r, agent_c) == (tr, tc):
        return GridWorld.PICK

    # Move greedily toward target
    if tr < agent_r:
        return GridWorld.UP
    if tr > agent_r:
        return GridWorld.DOWN
    if tc < agent_c:
        return GridWorld.LEFT
    if tc > agent_c:
        return GridWorld.RIGHT

    return GridWorld.UP
