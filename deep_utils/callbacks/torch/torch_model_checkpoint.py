import os

from deep_utils.utils.logging_utils import log_print
from deep_utils.utils.os_utils.os_path import split_extension


class ModelCheckPointTorch:
    def __init__(
        self,
        model_path,
        model,
        monitor="min",
        save_best_only=True,
        overwrite=True,
        save_last=True,
        loss=None,
        optimizer=None,
        scheduler=None,
        static_dict=None,
        monitor_val=None,
        logger=None,
        verbose=True,
    ):
        """

        :param model_path:
        :param model:
        :param monitor:
        :param save_best_only:
        :param overwrite:
        :param verbose:
        :param save_last:
        :param loss:
        :param optimizer:
        :param scheduler:
        :param static_dict:
        :param monitor_val: monitoring value. For starting the checkpointer from a saved model,
         by passing the monitoring value you preserve the best model.
        """
        self.overwrite = overwrite
        self.model_path = model_path
        self.monitor_val = float("-inf") if monitor == "max" else float("inf")
        self.monitor_val = self.monitor_val if monitor_val is None else monitor_val
        self.save_best_only = save_best_only
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.static_dict = static_dict
        self.monitor = monitor
        self.epoch = 0
        self.verbose = verbose
        self.save_last = save_last
        self.logger = logger
        os.makedirs(os.path.split(model_path)[0], exist_ok=True)
        log_print(
            self.logger,
            f"Initializing checkpointer with monitor-mode:{monitor} from monitor-val:{monitor_val}",
            verbose=verbose,
        )

    def __call__(self, monitor_val, **kwargs):
        self.epoch += 1
        if self.save_best_only:
            trigger = False
            if self.monitor == "min" and monitor_val < self.monitor_val:
                self.monitor_val = monitor_val
                trigger = True
            elif self.monitor == "max" and monitor_val > self.monitor_val:
                self.monitor_val = monitor_val
                trigger = True

            if self.save_last:
                last_path = split_extension(self.model_path, suffix="_last")
                self._save(last_path, print_=False,
                           monitor_val=monitor_val, **kwargs)

            if self.overwrite:
                best_path = split_extension(self.model_path, suffix="_best")
            else:
                best_path = split_extension(
                    self.model_path, suffix="_" + str(self.epoch)
                )

            if trigger:
                self._save(
                    best_path, print_=self.verbose, monitor_val=monitor_val, **kwargs
                )
        else:
            model_path = split_extension(
                self.model_path, suffix="_" + str(self.epoch))
            self._save(
                model_path, print_=self.verbose, monitor_val=monitor_val, **kwargs
            )

    def _save(self, model_path, print_, **extra):
        import torch

        save_dict = self.static_dict if self.static_dict is not None else dict()
        save_dict.update(extra if extra is not None else dict())
        save_dict["model_state_dict"] = self.model.state_dict()
        self._add_file(save_dict, "optimizer", self.optimizer.state_dict())
        self._add_file(save_dict, "scheduler", self.scheduler.state_dict())
        self._add_file(save_dict, "loss", self.loss)
        torch.save(save_dict, model_path)
        if print_:
            if "monitor_val" in extra:
                log_print(
                    self.logger,
                    f'monitor-val: {extra["monitor_val"]}, model is saved in {model_path}',
                    verbose=self.verbose,
                )
            else:
                log_print(
                    self.logger, f"model is saved in {model_path}", verbose=self.verbose
                )

    @staticmethod
    def _add_file(dict_, name, file):
        if file is not None:
            dict_[name] = file
        return dict_
