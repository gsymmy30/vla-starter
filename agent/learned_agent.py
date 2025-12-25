import torch
from models.policy import TinyVLAPolicy
from env.renderer import render_obs


class LearnedAgent:
    def __init__(self, vocab, checkpoint_path="policy.pt"):
        self.vocab = vocab
        self.model = TinyVLAPolicy(vocab)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.eval()

    def act(self, obs):
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

        # ----- Action masking (simple, teaches a real VLA concept) -----
        r, c = obs["agent_pos"]

        # infer grid size from coordinates in obs (agent + objects)
        coords = [tuple(obs["agent_pos"])] + [tuple(o["pos"]) for o in obs.get("objects", [])]
        max_rc = max(max(rr, cc) for rr, cc in coords)
        grid_size = max_rc + 1

        # boundary masks
        if r == 0:
            logits[0] = -1e9  # UP
        if r == grid_size - 1:
            logits[1] = -1e9  # DOWN
        if c == 0:
            logits[2] = -1e9  # LEFT
        if c == grid_size - 1:
            logits[3] = -1e9  # RIGHT

        # PICK is only valid if agent is on an object cell and not already holding
        on_object = any(tuple(o["pos"]) == (r, c) for o in obs.get("objects", []))
        if (not on_object) or (obs.get("holding") is not None):
            logits[4] = -1e9  # PICK

        # DROP only valid if holding something
        if obs.get("holding") is None:
            logits[5] = -1e9  # DROP

        return int(torch.argmax(logits).item())
