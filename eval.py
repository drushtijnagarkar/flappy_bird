from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQN
import torch
import time

# === Config ===
MODEL_PATH = "models/flappy_dqn.pt"
RENDER = True
EPISODES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load environment and model ===
env = FlappyBirdEnv(render_mode=RENDER)
model = DQN(input_dim=4, output_dim=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Run evaluation ===
for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        time.sleep(0.02)  # slow down for visibility

        state_tensor = torch.tensor([state], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            action = torch.argmax(model(state_tensor)).item()

        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

    print(f"Episode {episode+1}: Reward = {total_reward}")
