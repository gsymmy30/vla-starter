import json
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from env.renderer import render_obs


class VLADataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r") as f:
            self.episodes = json.load(f)

        self.samples = []

        # Flatten episodes into (obs, instruction, action) samples.
        # Oversample rare terminal actions like PICK so the policy learns to finish.
        self.samples = []
        for ep in self.episodes:
            for step in ep:
                self.samples.append(step)
                if step.get('action') == 4:  # PICK
                    for _ in range(10):
                        self.samples.append(step)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        obs = sample["obs"]

        # Render image on the fly
        img = render_obs(obs)
        img = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
             .view(img.size[1], img.size[0], 3)
             .numpy())
        ).float() / 255.0
        img = img.permute(2, 0, 1)  # C,H,W

        instruction = obs["instruction"]
        action = sample["action"]

        return img, instruction, torch.tensor(action, dtype=torch.long)