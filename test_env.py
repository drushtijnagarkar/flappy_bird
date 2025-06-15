from flappy_bird_env import FlappyBirdEnv
import time

# Initialize environment with rendering ON
env = FlappyBirdEnv(render_mode=True)

# Reset the game to start
state = env.reset()

# Run the game for a few steps
for step in range(500):
    # Example logic: flap every 20 steps
    action = 1 if step % 20 == 0 else 0

    # Step through environment with the selected action
    state, reward, done = env.step(action)

    # Show the game visually
    env.render()

    # Reset game if bird dies
    if done:
        print(f"Game Over. Resetting... (Score: {env.score})")
        time.sleep(1)
        state = env.reset()
