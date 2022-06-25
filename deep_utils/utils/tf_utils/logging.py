from deep_utils.utils.logging_utils import log_print


def save_train_val_figures(history, save_path, logger=None):
    import os
    from collections import defaultdict

    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)
    train_val = defaultdict(dict)
    for key, val in history.history.items():
        if key.startswith("val_"):
            train_val[key[4:]]["val"] = val
        else:
            train_val[key]["train"] = val
    for key, values in train_val.items():
        plt.figure(figsize=(10, 8))
        if "train" in values:
            plt.plot(values["train"], label="train")
        if "val" in values:
            plt.plot(values["val"], label="val")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel(key)
        plt.title(f"Metric {key}")
        plt.savefig(os.path.join(save_path, key + ".jpg"), dpi=500)
    log_print(logger=logger,
              message="Successfully saved figures to {save_path}!")
