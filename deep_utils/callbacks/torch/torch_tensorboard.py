from torch.utils.tensorboard import SummaryWriter
from deep_utils.utils.logging_utils.logging_utils import log_print


class TensorboardTorch:
    def __init__(
            self,
            log_dir,
            comment="",
            purge_step=None,
            max_queue=10,
            flush_secs=120,
            filename_suffix="",
            logger=None,
            verbose=0,
    ):
        """
        Creates an instance for TensorboardTorch class. Using this instance logs can be done easily.
        The required inputs at each call are an epoch number and an input key-values for logging.
        :param log_dir:
        :param comment:
        :param purge_step:
        :param max_queue:
        :param flush_secs:
        :param filename_suffix:
        :param logger:
        :param verbose:
        """
        self.writer = SummaryWriter(
            log_dir=log_dir,
            comment=comment,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )
        log_print(
            logger,
            f"Successfully created TensorboardTorch that saves to {log_dir}",
            verbose=verbose,
        )

    def __call__(self, epoch: int, **logs):
        """
        At each call a new record is created for the input key-value pair of inputs.
        :param epoch:
        :param logs:
        :return:
        """
        for metric_name, metric_value in logs.items():
            self.writer.add_scalar(metric_name, metric_value, epoch)
