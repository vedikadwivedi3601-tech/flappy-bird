import os

# Must be set before pygame/flappy_bird_gymnasium is imported, so it can
# render frames on a headless server (no physical display, like HF Spaces).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio
import gradio as gr
import torch
import flappy_bird_gymnasium
import gymnasium as gym

from dqn import DQN

device = "cpu"  # Spaces free tier is CPU-only; inference on this tiny net is instant anyway

MODEL_PATH = os.path.join("runs", "flappybirdv0.pt")


def play_episode(max_steps=2000):
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    policy_dqn = DQN(num_states, num_actions).to(device)
    policy_dqn.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy_dqn.eval()

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device)

    frames = [env.render()]
    total_reward = 0.0
    terminated = False
    steps = 0

    while not terminated and steps < max_steps:
        with torch.no_grad():
            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

        next_state, reward, terminated, _, _ = env.step(action.item())
        total_reward += reward

        frames.append(env.render())
        state = torch.tensor(next_state, dtype=torch.float, device=device)
        steps += 1

    env.close()

    out_path = "episode.mp4"
    imageio.mimsave(out_path, frames, fps=30)

    return out_path, f"Episode finished — steps: {steps}, total reward: {total_reward:.1f}"


with gr.Blocks(title="Flappy Bird DQN") as demo:
    gr.Markdown("# 🐦 Flappy Bird — Watch the trained DQN agent play")
    gr.Markdown("Click the button to run one episode with the trained model and watch it play back.")

    btn = gr.Button("▶ Watch AI play", variant="primary")
    video_out = gr.Video(label="Episode replay")
    status_out = gr.Textbox(label="Result", interactive=False)

    btn.click(fn=play_episode, inputs=None, outputs=[video_out, status_out])

if __name__ == "__main__":
    # Render (and most PaaS hosts) assign a port via the PORT env var and
    # expect the app to listen on 0.0.0.0, not just localhost.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
