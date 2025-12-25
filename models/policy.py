import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyVLAPolicy(nn.Module):
    def __init__(self, vocab: dict, num_actions: int = 6):
        super().__init__()
        self.vocab = vocab

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.text_embed = nn.Embedding(len(vocab), 16)

        self.fc = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def encode_text(self, text: str):
        tokens = text.lower().split()
        idxs = [self.vocab.get(t, 0) for t in tokens]
        t = torch.tensor(idxs, dtype=torch.long)
        emb = self.text_embed(t)
        return emb.mean(dim=0)

    def forward(self, img, instruction: str):
        img_feat = self.conv(img.unsqueeze(0)).view(-1)
        txt_feat = self.encode_text(instruction)
        feat = torch.cat([img_feat, txt_feat], dim=0)
        return self.fc(feat)
