import os
from env.gridworld import GridWorld
from env.renderer import render_obs
from agent.learned_agent import LearnedAgent
from models.dataset import VLADataset

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "PICK",
    5: "DROP",
}

def build_vocab_from_dataset(path):
    ds = VLADataset(path)
    vocab = {"<unk>": 0}
    for _, instr, _ in ds:
        for tok in instr.lower().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

def main():
    os.makedirs("rollout_frames", exist_ok=True)

    vocab = build_vocab_from_dataset("data/demo_trajectories.json")
    agent = LearnedAgent(vocab, checkpoint_path="policy.pt")

    env = GridWorld(size=7, seed=123)
    obs = env.reset("pick up the green block")

    print("instruction:", obs["instruction"])
    print("start:", "pos=", obs["agent_pos"], "objects=", obs["objects"])

    for t in range(25):
        # render BEFORE action
        render_obs(obs).save(f"rollout_frames/frame_{t:03d}.png")

        prev_pos = tuple(obs["agent_pos"])
        prev_holding = obs["holding"]

        action = agent.act(obs)
        name = ACTION_NAMES.get(action, str(action))

        obs, reward, done, info = env.step(action)

        changed = (tuple(obs["agent_pos"]) != prev_pos) or (obs["holding"] != prev_holding)

        print(
            f"t={t:02d} action={action} ({name}) "
            f"pos {prev_pos} -> {tuple(obs['agent_pos'])} "
            f"holding {prev_holding} -> {obs['holding']} "
            f"reward={reward:.2f} changed={changed}"
        )

        if done:
            render_obs(obs).save(f"rollout_frames/frame_{t+1:03d}.png")
            print("DONE:", info)
            break

    print("finished rollout")

if __name__ == "__main__":
    main()
