"""
This file defines the training function, and calls it
"""

from HH_Env import process_env
from Qlearning_agent import HHAgent


import time
from datetime import datetime

def train(agent, env, num_episodes=1000):
    score_per_ep = []
    start_time = time.time()

    # Start txt file with training info
    reward_log = open("reward_log.txt", "w")  # Open file
    reward_log.write("Training started at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
    
    # Training loop
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        state = tuple(obs.tolist())
        visited_states = set()

        
        while not done:
            action = agent.getAction(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = tuple(next_obs.tolist())
            
            # edited rewards due to non-improvement
            #if state not in visited_states:
                #reward += 1.0
                #visited_states.add(state)  # encourage exploration

            # Update the Q-values
            agent.update(state, action, next_state, reward)
            
            # Move to the next state
            state = next_state
            obs = next_obs
            total_reward += reward

            if truncated:
                break
    
        agent.decayEpsilon()  # Decay epsilon after each episode
        score_per_ep.append(total_reward)
        
        # Log every episode
        reward_log.write(f"Episode {ep + 1}: Score = {total_reward}\n")

        # Log average every 10 episodes
        if (ep + 1) % 10 == 0:
            avg_reward = sum(score_per_ep[-10:]) / 10
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"Average reward for episodes {ep + 1 - 10 + 1}-{ep + 1}: {avg_reward:.2f} at {timestamp}\n\n")
            reward_log.write(f"Average reward for episodes {ep + 1 - 10 + 1}-{ep + 1}: {avg_reward:.2f} at {timestamp}\n\n")

        reward_log.flush()

    # Update and close txt file
    total_time = time.time() - start_time  # End timing
    print(f"\nTime for completion {total_time:.2f}  seconds. ")
    reward_log.write(f"\nTraining completed in {total_time:.2f} seconds.\n")
    reward_log.close()


    return score_per_ep

# Run train()
if __name__ == "__main__":
    env = process_env()
    agent = HHAgent(
        env=env,
        learning_rate=0.1,
        initial_epsilon=1.0,
        epsilon_decay=0.995,
        final_epsilon=0.05,
        discount_factor=0.99
    )

    train(agent, env, num_episodes=1000)
