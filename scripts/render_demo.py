from env.gridworld import GridWorld
from env.renderer import render_obs

def main():
    env = GridWorld(size=7, seed=42)
    obs = env.reset("pick up the red block")
    img = render_obs(obs)
    img.save("out.png")
    print("Saved out.png")

if __name__ == "__main__":
    main()
