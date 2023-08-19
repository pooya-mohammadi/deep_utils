from collections import defaultdict
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from deep_utils.utils.logging_utils.logging_utils import log_print


class CSVLogger:
    def __init__(self, csv_path: Union[Path, str], logger=None, verbose=1):
        self.csv_path = csv_path
        self.logs: dict = defaultdict(list)
        log_print(
            logger,
            f"Successfully created CSVLogger that saves to {self.csv_path}",
            verbose=verbose,
        )

    def __call__(self, epoch=None, **logs):
        for key, val in logs.items():
            self.logs[key].append(val)
        self.__save()

    def __save(self):
        columns = list(self.logs.keys())
        values = np.array(list(self.logs.values())).T
        df = pd.DataFrame(values, columns=columns)
        df.to_csv(self.csv_path)
