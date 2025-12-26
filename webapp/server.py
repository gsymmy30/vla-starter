import base64
import io
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from env.gridworld import GridWorld
from env.renderer import render_obs
from agent.learned_agent import LearnedAgent
from models.dataset import VLADataset
from agent.expert import expert_action


app = FastAPI(title="vla-starter demo")

# Serve static files (our index.html)
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

# Global singleton state (simple, good for a demo)
ENV: Optional[GridWorld] = None
AGENT: Optional[LearnedAgent] = None
OBS = None
HISTORY = []  # list of (agent_pos, holding) for last few steps



def build_vocab_from_dataset(path: str):
    ds = VLADataset(path)
    vocab = {"<unk>": 0}
    for _, instr, _ in ds:
        for tok in instr.lower().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def obs_to_png_b64(obs) -> str:
    img = render_obs(obs)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ensure_init():
    global ENV, AGENT, OBS
    if ENV is None:
        ENV = GridWorld(size=7, seed=123)
    if AGENT is None:
        vocab = build_vocab_from_dataset("data/demo_trajectories.json")
        AGENT = LearnedAgent(vocab, checkpoint_path="policy.pt")
    if OBS is None:
        OBS = ENV.reset("pick up the green block")


@app.get("/", response_class=HTMLResponse)
def root():
    # Serve the static page
    with open("webapp/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/reset")
def api_reset(instruction: Optional[str] = None):
    ensure_init()
    global OBS
    if instruction is None or instruction.strip() == "":
        OBS = ENV.reset()
    else:
        OBS = ENV.reset(instruction.strip())

    return {
        "instruction": OBS["instruction"],
        "png_b64": obs_to_png_b64(OBS),
        "done": False,
        "info": {"step": 0, "holding": OBS.get("holding")},
    }


@app.post("/api/step")
def api_step():
    ensure_init()
    global OBS, HISTORY

    def sig(o):
        return (tuple(o["agent_pos"]), o.get("holding"))

    # choose an action from learned policy
    action = AGENT.act(OBS)

    prev_sig = sig(OBS)
    next_obs, reward, done, info = ENV.step(action)
    next_sig = sig(next_obs)

    # update history
    HISTORY.append(prev_sig)
    if len(HISTORY) > 6:
        HISTORY = HISTORY[-6:]

    # Detect simple 2-cycle oscillation: A,B,A,B
    oscillating = False
    if len(HISTORY) >= 4:
        a, b, c, d = HISTORY[-4], HISTORY[-3], HISTORY[-2], HISTORY[-1]
        if a == c and b == d and a != b:
            oscillating = True

    # Recovery:
    # 1) if we are oscillating, fall back to the expert for one step
    # 2) if action was somehow a no-op, also fall back once
    if oscillating or (next_sig == prev_sig):
        action = expert_action(ENV, OBS)
        next_obs, reward, done, info = ENV.step(action)

        # refresh history after recovery step
        HISTORY.append(prev_sig)
        if len(HISTORY) > 6:
            HISTORY = HISTORY[-6:]

    OBS = next_obs

    return {
        "action": int(action),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
        "png_b64": obs_to_png_b64(OBS),
        "instruction": OBS["instruction"],
    }


    action = AGENT.act(OBS)
    next_obs, reward, done, info = ENV.step(action)
    OBS = next_obs

    return {
        "action": int(action),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
        "png_b64": obs_to_png_b64(OBS),
        "instruction": OBS["instruction"],
    }
