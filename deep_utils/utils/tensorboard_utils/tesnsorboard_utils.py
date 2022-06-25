import traceback
from os.path import join

import numpy as np


class TensorboardUtils:
    @staticmethod
    def save_image(df, path):
        import matplotlib.pyplot as plt
        from pandas.plotting import table

        ax = plt.subplot(111, frame_on=False)  # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        table(ax, df)  # where df is your data frame
        plt.savefig(join(path, "csv_log.png"))

    @staticmethod
    def get_docx(df, doc_path):
        import docx

        # open an existing document
        doc = docx.Document(None)

        # add a table to the end and create a reference variable
        # extra row is so we can add the header row
        t = doc.add_table(df.shape[0] + 1, df.shape[1])

        # add the header rows.
        for j in range(df.shape[-1]):
            t.cell(0, j).text = df.columns[j]

        # add the rest of the data frame
        for i in range(df.shape[0]):
            for j in range(df.shape[-1]):
                t.cell(i + 1, j).text = str(df.values[i, j])

        # save the doc
        doc.save(doc_path)

    @staticmethod
    def save_df(path, file_name=None):
        import pandas as pd
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        try:
            if file_name is not None:
                event_acc = EventAccumulator(join(path, file_name))
            else:
                event_acc = EventAccumulator(path)
            event_acc.Reload()
            tags = event_acc.Tags()["scalars"]
            values_df, columns = [], []
            for tag in tags:
                if tag == "epoch":
                    continue
                event_list = event_acc.Scalars(tag)
                values = list(map(lambda x: x.value, event_list))
                if "elapsed_time" not in columns:
                    wall_time = list(map(lambda x: x.wall_time, event_list))
                    wall_time = (
                        np.array(wall_time[1:]) - np.array(wall_time[:-1])
                    ).tolist()
                    wall_time = [np.mean(wall_time)] + wall_time
                    values_df.extend([values, wall_time])
                    columns.extend([f"{tag}", f"elapsed_time"])
                else:
                    values_df.append(values)
                    columns.append(tag)
        except Exception:
            print("Event file possibly corrupt: {}".format(path))
            traceback.print_exc()
        # appending epoch
        values_df.insert(0, list(range(1, len(values_df[0]) + 1)))
        columns.insert(0, "epoch")
        values = np.round(np.array(values_df).T, 3)
        df = pd.DataFrame(values, columns=columns)
        df = df[
            [
                "epoch",
                "elapsed_time",
                "train_f1_score",
                "train_loss",
                "train_acc",
                "val_f1_score",
                "val_loss",
                "val_acc",
            ]
        ]
        df_path = join(path, "csv_log.csv")
        df.to_csv(df_path, index=False)
        if "elapsed_time" in df:
            print("mean time:", df["elapsed_time"].mean(axis=0))
            print("sum time:", df["elapsed_time"].sum(axis=0))

        print(f"[INFO] Successfully saved results to {df_path}")
        return df


if __name__ == "__main__":
    path = (
        "/home/ai/projects/car_recognition/output/exp_1/train/lightning_logs/version_0"
    )
    TensorboardUtils.save_df(path)
