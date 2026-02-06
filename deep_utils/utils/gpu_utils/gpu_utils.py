import subprocess
from io import StringIO


class GPUUtils:

    @staticmethod
    def get_free_gpu(verbose:bool=True):
        import pandas as pd
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        gpu_df = pd.read_csv(StringIO(gpu_stats.decode()),
                             names=['memory.used', 'memory.free'],
                             skiprows=1)
        if verbose:
            print('GPU usage:\n{}'.format(gpu_df))
        gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
        idx = gpu_df['memory.free'].idxmax()
        if verbose:
            print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
        return idx
