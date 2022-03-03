from deep_utils.utils.utils.logging_ import log_print


class TensorboardTorch:
    def __init__(self, log_dir, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix='', logger=None, verbose=1):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment, purge_step=purge_step, max_queue=max_queue,
                                    flush_secs=flush_secs, filename_suffix=filename_suffix)
        log_print(logger, f"Successfully created TensorboardTorch that saves to {log_dir}", verbose=verbose)

    def __call__(self, epoch: int, **logs):
        for metric_name, metric_value in logs.items():
            self.writer.add_scalar(metric_name, metric_value, epoch)
