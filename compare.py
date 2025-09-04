import os
import numpy as np
import matplotlib.pyplot as plt
from env import McDonaldsEnv
from agent import FastDQNAgent

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPECTED_SERVICE = {0: 1.0, 1: 0.1, 2: 0.1}


def run_episode(env, policy, agent=None, seed=None, max_steps=2000):
    env.reset()
    if seed is not None:
        env.rng.seed(seed)
    done = False
    steps = 0
    while not done and steps < max_steps:
        state = env._get_state()

        if int(state[0] + state[1] + state[2]) == 0:
            if env.env.peek() is None or env.env.now >= env.sim_time:
                break
            env.env.step()
            continue

        eligible = env._eligible_indices()
        if not eligible:
            env.env.step()
            continue

        if policy == 'fcfs':
            idx = eligible[0]
            _, r, done, _ = env.serve_order_index(idx)
        elif policy == 'spt':
            best_i = eligible[0]
            best_est = float('inf')
            for i in eligible:
                order = env.queue[i]
                est = sum(EXPECTED_SERVICE[t] for t in order.items)
                if est < best_est:
                    best_est = est
                    best_i = i
            _, r, done, _ = env.serve_order_index(best_i)
        elif policy == 'rl':
            mask = np.zeros(agent.action_dim, dtype=bool)
            for i in eligible:
                mask[i] = True
            a = agent.act_masked(state, mask)
            _, r, done, _ = env.serve_order_index(a)
        else:
            raise ValueError('unknown policy')

        steps += 1

    avg_wait = (env.total_wait / env.served_orders) if env.served_orders > 0 else 0.0
    rejected = getattr(env, 'rejected_orders', 0)
    return avg_wait, env.served_orders, rejected


def evaluate(policies=('fcfs', 'spt', 'rl'), episodes=200, seed=0):
    env = McDonaldsEnv(seed=seed, sim_time=8*60, arrival_rate=1/3.0, max_active_orders=20, serve_window=3)
    agent = FastDQNAgent(state_dim=6, action_dim=20)
    model_path = 'models/fastdqn_final.npz'
    if os.path.exists(model_path):
        print(f'Loading RL agent weights from {model_path}')
        agent.load(model_path)
    else:
        print('No trained RL model found, using untrained agent.')


    results = {p: [] for p in policies}
    served = {p: [] for p in policies}
    rejected = {p: [] for p in policies}

    for p in policies:
        for ep in range(episodes):
            avg_wait, nserved, nrej = run_episode(env, p, agent=agent if p == 'rl' else None, seed=seed+ep)
            results[p].append(avg_wait)
            served[p].append(nserved)
            rejected[p].append(nrej)


    labels = list(policies)
    data = [results[p] for p in labels]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel('Average wait per order (minutes)')
    plt.title(f'Policy comparison (n={episodes})')
    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, 'policy_boxplot.png')
    plt.savefig(out1)
    print('Saved', out1)


    for p in labels:
        print(f"{p}: mean_wait={np.mean(results[p]):.3f}, std_wait={np.std(results[p]):.3f}, served_orders_mean={np.mean(served[p]):.1f}")

    return results, served


if __name__ == '__main__':
    evaluate(policies=('fcfs', 'spt', 'rl'), episodes=600)
