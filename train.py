from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import torch
import matplotlib.pyplot as plt

# === Settings ===
EPISODES = 1000
RENDER = False  # Set to True to visualize training
SAVE_PATH = "models/flappy_dqn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Init ===
env = FlappyBirdEnv(render_mode=RENDER)
agent = DQNAgent(state_dim=4, action_dim=2, device=DEVICE)
all_rewards = []

# === Training Loop ===
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        if RENDER:
            env.render()

        action = agent.act(state)
        next_state, reward, done = env.step(action)

        agent.push(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state
        total_reward += reward

    all_rewards.append(total_reward)
    print(f"Episode {episode+1}/{EPISODES}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    # Save model every 50 episodes
    if (episode + 1) % 50 == 0:
        torch.save(agent.policy_net.state_dict(), SAVE_PATH)
        
best_reward = float('-inf')

# Inside training loop after episode ends
if total_reward > best_reward:
    best_reward = total_reward
    torch.save(agent.policy_net.state_dict(), SAVE_PATH)
    print(f"✅ New best model saved with reward {total_reward}")


# === Plot Rewards ===
plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress")
plt.savefig("plots/training_reward.png")
plt.show()
