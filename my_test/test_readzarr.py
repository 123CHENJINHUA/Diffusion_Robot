import zarr
import numpy as np
input_name = 'test_record/replay_buffer.zarr'
dataset_name = 'data/action'
f = zarr.open(input_name)
raw = f[dataset_name ]
print(raw.shape)

