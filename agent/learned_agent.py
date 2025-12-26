import torch
from models.policy import TinyVLAPolicy
from env.renderer import render_obs


ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_PICK = 4
ACTION_DROP = 5


class LearnedAgent:
    def __init__(self, vocab, checkpoint_path="policy.pt"):
        self.vocab = vocab
        self.model = TinyVLAPolicy(vocab)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.eval()

    def act(self, obs):
        # Render observation to image tensor
        img = render_obs(obs)
        img = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
             .view(img.size[1], img.size[0], 3)
             .numpy())
        ).float() / 255.0
        img = img.permute(2, 0, 1)

        instruction = obs["instruction"]

        with torch.no_grad():
            logits = self.model(img, instruction).clone()

        r, c = tuple(obs["agent_pos"])

        # Infer grid size from coords present (agent + objects)
        coords = [tuple(obs["agent_pos"])] + [tuple(o["pos"]) for o in obs.get("objects", [])]
        max_rc = max(max(rr, cc) for rr, cc in coords) if coords else 0
        grid_size = int(max_rc + 1)

        # Boundary masks: prevent actions that won't change state
        if r == 0:
            logits[ACTION_UP] = -1e9
        if r == grid_size - 1:
            logits[ACTION_DOWN] = -1e9
        if c == 0:
            logits[ACTION_LEFT] = -1e9
        if c == grid_size - 1:
            logits[ACTION_RIGHT] = -1e9

        # PICK only if standing on an object and not holding already
        on_object = any(tuple(o["pos"]) == (r, c) for o in obs.get("objects", []))
        if (not on_object) or (obs.get("holding") is not None):
            logits[ACTION_PICK] = -1e9

        # DROP only if holding something
        if obs.get("holding") is None:
            logits[ACTION_DROP] = -1e9

        return int(torch.argmax(logits).item())
