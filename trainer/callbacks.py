# trainer/callbacks.py


class BaseCallback:
    """
    A generic callback interface. Subclass and override any methods you need.
    """

    def on_training_start(self, trainer):
        pass

    def on_training_step(self, trainer, step_data):
        pass

    def on_episode_end(self, trainer, episode_data):
        pass

    def on_training_end(self, trainer):
        pass


class CheckpointCallback(BaseCallback):
    """
    Saves the model every 'save_freq' episodes.
    """

    def __init__(self, save_freq=10, save_path="./checkpoint.pkl"):
        self.save_freq = save_freq
        self.save_path = save_path

    def on_episode_end(self, trainer, episode_data):
        if (trainer.episode_count % self.save_freq) == 0:
            print(
                f"[CheckpointCallback] Saving model at episode {trainer.episode_count}"
            )
            trainer.model.save_checkpoint(self.save_path)


class LoggingCallback(BaseCallback):
    """
    Logs step rewards and losses. You can extend this for TensorBoard, Weights & Biases, etc.
    """

    def __init__(self, log_file="training_log.csv"):
        self.log_file = log_file
        # Write CSV header
        with open(self.log_file, "w") as f:
            f.write("episode,total_reward,loss\n")

    def on_episode_end(self, trainer, episode_data):
        episode = trainer.episode_count
        total_reward = episode_data["total_reward"]
        loss = episode_data.get("loss", None)
        with open(self.log_file, "a") as f:
            f.write(f"{episode},{total_reward},{loss}\n")
