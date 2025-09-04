import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FastDQNAgent:
    def __init__(self, state_dim=6, action_dim=10, lr=1e-3, gamma=0.99,
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01,
                 memory_size=5000, batch_size=32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model

    def act(self, state, mask=None):
        if np.random.rand() < self.epsilon:
            if mask is not None and np.any(mask):
                eligible = np.where(mask)[0]
                return int(np.random.choice(eligible))
            return int(np.random.choice(self.action_dim))
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        q = self.model.predict_on_batch(state)[0]
        if mask is not None and np.any(mask):
            q_masked = np.where(mask, q, -1e9)
            return int(np.argmax(q_masked))
        return int(np.argmax(q))

    def action_probs(self, state, temperature=1.0):
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        q = self.model.predict_on_batch(state)[0]
        q = q - np.max(q)
        exps = np.exp(q / max(1e-6, temperature))
        return exps / np.sum(exps)

    def act_masked(self, state, mask):
        return self.act(state, mask)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.asarray(state, dtype=np.float32), int(action), float(reward), np.asarray(next_state, dtype=np.float32), bool(done)))

    def sample_batch(self, batch_size=None):
        bs = batch_size or self.batch_size
        if len(self.memory) < bs:
            return None
        batch = random.sample(self.memory, bs)
        states = np.vstack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.vstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        return states, actions, rewards, next_states, dones

    def train_from_memory(self, batch_size=None):
        sample = self.sample_batch(batch_size)
        if sample is None:
            return
        states, actions, rewards, next_states, dones = sample
        q_pred = self.model.predict_on_batch(states)
        q_next = self.model.predict_on_batch(next_states)
        max_next = np.max(q_next, axis=1)
        targets = rewards + (1.0 - dones) * self.gamma * max_next
        q_target_mat = q_pred.copy()
        q_target_mat[np.arange(states.shape[0]), actions] = targets
        self.model.train_on_batch(states, q_target_mat)

    def replay(self):
        self.train_from_memory()

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path):
        if path.endswith('.npz'):
            weights = self.model.get_weights()
            np.savez(path, *weights)
        else:
            self.model.save_weights(path)

    def load(self, path):
        if path.endswith('.npz'):
            data = np.load(path)
            def sort_key(x):
                if x.startswith('arr_'):
                    try:
                        return int(x.split('_')[1])
                    except Exception:
                        return x
                return x
            weights = [data[k] for k in sorted(data.files, key=sort_key)]
            if not weights:
                weights = [data[f] for f in data.files]
            self.model.set_weights(weights)
        else:
            self.model.load_weights(path)
