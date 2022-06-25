import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class LRScalar(Callback):
    def __init__(self, round_n: int = 6):
        super().__init__()
        self.global_step = 0
        self.round_n = round_n

    def on_batch_end(self, batch, logs=None):
        try:
            lr = round(self.model.optimizer.lr.numpy(), self.round_n)
        except AttributeError:
            lr = round(
                float(self.model.optimizer.lr(
                    self.global_step).numpy()), self.round_n
            )
        tf.summary.scalar(name="learning_rate", data=lr, step=self.global_step)
        self.global_step += 1
