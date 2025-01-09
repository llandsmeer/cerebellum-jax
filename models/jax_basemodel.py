# models/jax_cartpole_model.py

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from typing import Sequence

from models.base_model import BaseModel


class CartpolePolicy(nn.Module):
    """
    Simple MLP policy for CartPole with discrete actions = {0, 1}.
    """

    hidden_sizes: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_sizes:
            x = nn.relu(nn.Dense(size)(x))
        # Output is logits for discrete actions
        logits = nn.Dense(2)(x)
        return logits


class JAXCartpoleModel(BaseModel):
    def __init__(self, obs_dim: int, hidden_sizes=(64, 64), lr=1e-3, seed=42):
        """
        Initialize the model, optimizer, and random keys.
        """
        # Define the neural network
        self.net = CartpolePolicy(hidden_sizes=hidden_sizes)

        # Random key for parameter init
        self.rng = jax.random.PRNGKey(seed)

        # Initialize parameters (Flax)
        dummy_obs = jnp.ones((1, obs_dim))
        self.params = self.net.init(self.rng, dummy_obs)

        # Create the optimizer
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

    def predict(self, obs):
        """
        For a single observation or a batch of observations (shape: [batch_size, obs_dim]),
        return the greedy action according to the policy logits.
        """
        logits = self.net.apply(self.params, obs)
        # Action = argmax of logits
        action = jnp.argmax(logits, axis=-1)
        return action

    def train_step(self, batch):
        """
        A single gradient update step.
        Batch is a dictionary with arrays for 'obs', 'actions', 'rewards', 'next_obs', 'dones'.
        Here we'll do a minimal cross-entropy-like update for demonstration.
        """
        obs = batch["obs"]
        actions = batch["actions"]

        def loss_fn(params):
            logits = self.net.apply(params, obs)
            # A simple supervised-like cross-entropy:
            one_hot_actions = jax.nn.one_hot(actions, 2)
            log_probs = jax.nn.log_softmax(logits)
            loss = -jnp.mean(jnp.sum(one_hot_actions * log_probs, axis=-1))
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(self.params)
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, self.params
        )
        self.params = optax.apply_updates(self.params, updates)

        return {"loss": loss}

    def save_checkpoint(self, path):
        """
        Save model parameters.
        """
        import pickle

        checkpoint = {"params": self.params, "opt_state": self.opt_state}
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path):
        """
        Load model parameters.
        """
        import pickle

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        self.params = checkpoint["params"]
        self.opt_state = checkpoint["opt_state"]
