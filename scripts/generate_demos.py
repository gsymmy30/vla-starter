import json
from env.gridworld import GridWorld
from agent.expert import expert_action
from env.renderer import render_obs


def run_episode(env: GridWorld):
    obs = env.reset()
    trajectory = []

    done = False
    while not done:
        action = expert_action(env, obs)
        next_obs, reward, done, info = env.step(action)

        trajectory.append({
            "obs": obs,
            "action": action,
            "reward": reward,
        })

        obs = next_obs

    return trajectory


def main():
    env = GridWorld(size=7, seed=0)
    demos = []

    for _ in range(20):
        traj = run_episode(env)
        demos.append(traj)

    with open("data/demo_trajectories.json", "w") as f:
        json.dump(demos, f, indent=2)

    # Render last frame for sanity check
    img = render_obs(obs=traj[-1]["obs"])
    img.save("demo_last_frame.png")

    print(f"Saved {len(demos)} trajectories to data/demo_trajectories.json")


if __name__ == "__main__":
    main()
