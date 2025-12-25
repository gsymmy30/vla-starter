import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models.dataset import VLADataset
from models.policy import TinyVLAPolicy


def build_vocab(dataset):
    vocab = {"<unk>": 0}
    for img, instr, act in dataset:
        for tok in instr.lower().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def main():
    dataset = VLADataset("data/demo_trajectories.json")
    vocab = build_vocab(dataset)

    model = TinyVLAPolicy(vocab)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(5):
        total_loss = 0.0
        for img, instr, action in loader:
            logits = model(img[0], instr[0])
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), action
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"epoch {epoch} loss {total_loss:.3f}")

    torch.save(model.state_dict(), "policy.pt")
    print("saved policy.pt")


if __name__ == "__main__":
    main()
