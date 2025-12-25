from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

Action = int  # 0..5


@dataclass
class Obj:
    color: str
    pos: Tuple[int, int]  # (row, col)


class GridWorld:
    """
    Minimal embodied environment:
    - State is a grid with colored objects + an agent.
    - Observation will be an RGB image produced by env/renderer.py.
    - Actions are discrete: up, down, left, right, pick, drop.
    """

    # Action IDs
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PICK = 4
    DROP = 5

    def __init__(
        self,
        size: int = 7,
        max_steps: int = 50,
        seed: int = 0,
    ) -> None:
        assert size >= 5, "size should be at least 5"
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.agent_pos: Tuple[int, int] = (0, 0)
        self.objects: List[Obj] = []
        self.holding: Optional[Obj] = None
        self.step_count: int = 0
        self.instruction: str = ""

    def reset(self, instruction: Optional[str] = None) -> Dict:
        self.step_count = 0
        self.holding = None

        # Place agent away from borders to make movement interesting
        self.agent_pos = (self.rng.integers(1, self.size - 1), self.rng.integers(1, self.size - 1))

        # Place a few objects with unique colors
        colors = ["red", "blue", "green"]
        self.objects = []
        taken = {self.agent_pos}
        for c in colors:
            pos = self._sample_empty_cell(taken)
            taken.add(pos)
            self.objects.append(Obj(color=c, pos=pos))

        if instruction is None:
            # Default instruction: pick up a random colored object
            target = self.rng.choice(colors)
            self.instruction = f"pick up the {target} block"
        else:
            self.instruction = instruction

        return self._get_obs()

    def step(self, action: Action) -> Tuple[Dict, float, bool, Dict]:
        self.step_count += 1
        reward = 0.0
        done = False

        if action in (self.UP, self.DOWN, self.LEFT, self.RIGHT):
            self._move(action)
        elif action == self.PICK:
            picked = self._pick()
            if picked:
                reward += 0.1
        elif action == self.DROP:
            dropped = self._drop()
            if dropped:
                reward += 0.05
        else:
            raise ValueError(f"Unknown action: {action}")

        # Simple success condition: if instruction is "pick up the X block", success when holding X
        target_color = self._parse_pick_target(self.instruction)
        if target_color is not None and self.holding is not None and self.holding.color == target_color:
            reward += 1.0
            done = True

        if self.step_count >= self.max_steps:
            done = True

        info = {
            "step": self.step_count,
            "holding": None if self.holding is None else self.holding.color,
        }
        return self._get_obs(), reward, done, info

    def _move(self, action: Action) -> None:
        r, c = self.agent_pos
        if action == self.UP:
            r -= 1
        elif action == self.DOWN:
            r += 1
        elif action == self.LEFT:
            c -= 1
        elif action == self.RIGHT:
            c += 1

        r = int(np.clip(r, 0, self.size - 1))
        c = int(np.clip(c, 0, self.size - 1))
        self.agent_pos = (r, c)

        # If holding an object, it moves with the agent
        if self.holding is not None:
            self.holding.pos = self.agent_pos

    def _pick(self) -> bool:
        if self.holding is not None:
            return False
        for obj in self.objects:
            if obj.pos == self.agent_pos:
                self.holding = obj
                return True
        return False

    def _drop(self) -> bool:
        if self.holding is None:
            return False
        # Dropping means "keep object at current agent position and release"
        self.holding.pos = self.agent_pos
        self.holding = None
        return True

    def _sample_empty_cell(self, taken: set) -> Tuple[int, int]:
        while True:
            pos = (int(self.rng.integers(0, self.size)), int(self.rng.integers(0, self.size)))
            if pos not in taken:
                return pos

    def _parse_pick_target(self, instruction: str) -> Optional[str]:
        s = instruction.lower()
        if "pick up the" in s and "block" in s:
            # naive parse: "pick up the {color} block"
            parts = s.split()
            try:
                idx = parts.index("the")
                color = parts[idx + 1]
                return color
            except Exception:
                return None
        return None

    def _get_obs(self) -> Dict:
        """
        Observation is a dict so we can later add:
        - image
        - text instruction
        - metadata

        IMPORTANT: keep this JSON-serializable (no numpy types).
        """
        ar, ac = self.agent_pos
        return {
            "instruction": self.instruction,
            "agent_pos": (int(ar), int(ac)),
            "objects": [
                {"color": o.color, "pos": (int(o.pos[0]), int(o.pos[1]))}
                for o in self.objects
            ],
            "holding": None if self.holding is None else self.holding.color,
        }
