import sys
sys.path.insert(0,sys.path[0]+'/../')
import multiprocessing as mp
import numpy as np
import time
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from multiprocessing.managers import SharedMemoryManager


import spacenavigator_linux as spacenavigator

class Spacemouse(mp.Process):
    def __init__(self, 
            shm_manager, 
            get_max_k=30, 
            frequency=200,
            max_value=500, 
            deadzone=(0,0,0,0,0,0), 
            dtype=np.float32,
            n_buttons=2,
            is_2D_motion = False
            ):
        """
        Continuously listen to 3D connection space naviagtor events
        and update the latest state.

        max_value: {300, 500} 300 for wired version and 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0
        
        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        super().__init__()
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        # copied variables
        self.is_2D_motion = is_2D_motion
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons
        # self.motion_event = SpnavMotionEvent([0,0,0], [0,0,0], 0)
        # self.button_state = defaultdict(lambda: False)
        self.xyz_spnav = np.array([
            [1.5,0,0],
            [0,-1.5,0],
            [0,0,-1.5]
        ], dtype=dtype)

        self.angle_spnav = np.array([
            [0,-0.2,0],
            [-0.2,0,0],
            [0,0,0.2]
        ], dtype=dtype)

        example = {
            # 3 translation, 3 rotation, 1 period
            # 'motion_event': np.zeros((7,), dtype=np.int64), create menmory buffer

            'motion_event': np.zeros((7,), dtype=np.float64),
            # left and right button
            'button_state': np.zeros((n_buttons,), dtype=bool),
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

    # ======= get state APIs ==========

    # def get_motion_state(self):
    #     state = self.ring_buffer.get()
    #     state = np.array(state['motion_event'][:6], 
    #         dtype=self.dtype) / self.max_value
    #     is_dead = (-self.deadzone < state) & (state < self.deadzone)
    #     state[is_dead] = 0
    #     return state

    def get_motion_state(self):
        state = self.ring_buffer.get()
        state = np.array(state['motion_event'][:6], 
            dtype=self.dtype)
        return state
    
    def get_motion_state_transformed(self):
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back

        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.xyz_spnav @ state[:3]
        tf_state[3:] = self.angle_spnav @ state[3:]
        return tf_state

    def get_button_state(self):
        state = self.ring_buffer.get()
        return state['button_state']
    
    def is_button_pressed(self, button_id):
        return self.get_button_state()[button_id]
    
    #========== start stop API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop ==========
    def run(self):
        success = spacenavigator.open(frequency=self.frequency)
        print("sucess:{}".format(success.connected))
        try:
            # motion_event = np.zeros((7,), dtype=np.int64)
            motion_event = np.zeros((7,), dtype=np.float64)

            button_state = np.zeros((self.n_buttons,), dtype=bool)
            # send one message immediately so client can start reading
            self.ring_buffer.put({
                'motion_event': motion_event,
                'button_state': button_state,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()

            while not self.stop_event.is_set():
                event = spacenavigator.read()
                receive_timestamp = time.time()

                if self.is_2D_motion:
                    translation = [event.x, event.y, 0]
                    rotation = [0, 0, 0]

                else:
                    translation = [event.x, event.y, event.z]
                    rotation = [event.roll, event.pitch, event.yaw]
                
                motion_event[:3] = translation
                motion_event[3:6] = rotation
                motion_event[6] = event.t
                
                # print(motion_event)
                button_state = event.buttons
                # print(button_state)
            
                # finish integrating this round of events
                # before sending over
                self.ring_buffer.put({
                    'motion_event': motion_event,
                    'button_state': button_state,
                    'receive_timestamp': receive_timestamp
                })
                time.sleep(1/self.frequency)
        finally:
            spacenavigator.close()

if __name__ == '__main__':
    
    with SharedMemoryManager() as shm_manager:
        sm = Spacemouse(shm_manager=shm_manager,is_2D_motion = False)
        sm.start()

        while True:
            # state = Sp.ring_buffer.get()
            # print(state['motion_event'])

            sm_state = sm.get_motion_state_transformed()
            # sm_press = sm.is_button_pressed(0)
            print(sm_state)





