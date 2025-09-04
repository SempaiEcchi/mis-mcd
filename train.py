import os
import numpy as np
from env import McDonaldsEnv
from agent import FastDQNAgent

os.makedirs('models', exist_ok=True)

def train(episodes=500, seed=0):
    env = McDonaldsEnv(seed=seed, sim_time=8*60, arrival_rate=1/3.0, max_active_orders=20)
    agent = FastDQNAgent(state_dim=6, action_dim=20, epsilon_decay=0.999, batch_size=64, memory_size=10000)

    for ep in range(1, episodes+1):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 2000:
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            steps += 1


        for _ in range(20):
            agent.train_from_memory()

        agent.update_epsilon()

        avg_wait = (env.total_wait / env.served_orders) if env.served_orders > 0 else 0.0
        print(f"Episode {ep:03d}: served_orders={env.served_orders}, avg_wait={avg_wait:.3f} min, steps={steps}")

    try:
        agent.save('models/fastdqn_final.npz')
        print('Final model saved to models/fastdqn_final.npz')
    except Exception as e:
        print('Failed to save final model:', e)

if __name__ == '__main__':
    train(episodes=1000)
