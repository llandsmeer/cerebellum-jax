# trainer/trainer.py

import numpy as np
import gymnasium as gym


class Trainer:
    def __init__(
        self,
        model,
        env,
        callbacks=None,
        gamma=0.99,
        max_episodes=1000,
        max_steps_per_episode=500,
    ):
        """
        :param model: A subclass of BaseModel
        :param env: A gymnasium environment
        :param callbacks: List of callback instances
        :param gamma: Discount factor
        :param max_episodes: Max episodes to train
        :param max_steps_per_episode: Max steps in an episode before termination
        """
        self.model = model
        self.env = env
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode

        self.episode_count = 0
        self.step_count = 0

        self.callbacks = callbacks if callbacks else []

    def train(self):
        for cb in self.callbacks:
            cb.on_training_start(self)

        for episode_idx in range(self.max_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            transitions = []

            for step in range(self.max_steps_per_episode):
                # Convert obs to jnp if using JAX model
                # But if your model supports np arrays, that’s fine
                action = np.array(self.model.predict(obs[None, :]))[0]
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                transitions.append((obs, action, reward, next_obs, done))

                obs = next_obs
                total_reward += reward
                self.step_count += 1

                # If episode is done, break
                if done:
                    break

            # Prepare a single batch from transitions (on-policy for demonstration)
            # We can pass everything in arrays
            batch = {
                "obs": np.array([t[0] for t in transitions]),
                "actions": np.array([t[1] for t in transitions]),
                "rewards": np.array([t[2] for t in transitions]),
                "next_obs": np.array([t[3] for t in transitions]),
                "dones": np.array([t[4] for t in transitions]),
            }

            # Do a single training step (in real RL, you'd accumulate transitions in a buffer or compute returns, etc.)
            # but let's keep it simple here
            output = self.model.train_step(batch)
            loss = output.get("loss", None)

            self.episode_count += 1

            # Call callbacks
            episode_data = {
                "episode_idx": episode_idx,
                "total_reward": total_reward,
                "loss": float(loss) if loss is not None else None,
            }
            for cb in self.callbacks:
                cb.on_episode_end(self, episode_data)

        for cb in self.callbacks:
            cb.on_training_end(self)

    def visualize(self, num_episodes=3):
        """
        Run a few episodes with the current policy and render them.
        """
        for i in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = np.array(self.model.predict(obs[None, :]))[0]
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Render
                self.env.render()

            print(f"[Visualize] Episode {i+1}, Total Reward: {total_reward}")
        self.env.close()
