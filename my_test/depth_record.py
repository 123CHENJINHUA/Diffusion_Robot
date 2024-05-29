import numpy as np
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs


class Depth_record:
    def __init__(self,fps):
        self.fps = fps
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0

    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.container is not None

    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0

    def start(self,file_path,start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()
            
        self.container = []
        self.start_time = start_time
        self.file_path = file_path

    def write_frame(self, depth_data: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError('Must run start() before writing!')
        if self.start_time is not None:
            local_idxs, global_idxs, self.next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1/self.fps,
                next_global_idx=self.next_global_idx
            )

        # number of appearance means repeats
        n_repeats = len(local_idxs)
        self.container.append(depth_data.tolist())

    def stop(self):
        if not self.is_ready():
            return
        
        save_data = np.array(self.container)
        np.save(self.file_path,save_data)
