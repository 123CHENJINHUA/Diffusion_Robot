import sys
sys.path.insert(0,sys.path[0]+'/../')

import hid
import time
import numpy as np
import timeit
from collections import namedtuple

import multiprocessing as mp
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from multiprocessing.managers import SharedMemoryManager

# clock for timing
high_acc_clock = timeit.default_timer

def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.

    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte

    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.

    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling

    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x
def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.

    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte

    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))

SpaceNavigator = namedtuple(
    "SpaceNavigator", ["t", "x", "y", "z", "roll", "pitch", "yaw", "buttons"]
)

class ButtonState(list):
    def __int__(self):
        return sum((b << i) for (i, b) in enumerate(reversed(self)))

class DeviceSpec(mp.Process):

    def __init__(self,shm_manager,vendor_id,product_id,axis_scale=350.0, min_v=-1.0, max_v=1.0,frequency = 60):
        
        super().__init__()

        self.vendor_id = vendor_id
        self.product_id = product_id
        self.axis_scale = axis_scale
        self.min_v = min_v
        self.max_v = max_v
        self.dict_state = {
            "t": -1,
            "x": 0,
            "y": 0,
            "z": 0,
            "roll": 0,
            "pitch": 0,
            "yaw": 0,
            "buttons": np.array(ButtonState([0] * 2)),
        }

        self.tuple_state = SpaceNavigator(**self.dict_state)

        # build input queue
        example = {
            "t": -1,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "buttons": np.zeros((2,),dtype=np.int16),

        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=1024#256
        )


        # start in disconnected state
        self.device = None
        self.callback = None
        self.button_callback = None
        self.all_recv = [0,0]
        self.frequency = frequency

        self.input_queue = input_queue

    @property
    def connected(self):
        """True if the device has been connected"""
        return self.device is not None

    @property
    def state(self):
        """Return the current value of read()

        Returns: state: {t,x,y,z,pitch,yaw,roll,button} namedtuple
                None if the device is not open.
        """
        return self.read()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def open(self):
        """Open a connection to the device, if possible"""
        if self.device is None:
            self.device = hid.device()
            self.device.open(self.vendor_id, self.product_id)
        
    def close(self):
        """Close the connection, if it is open"""
        if self.connected:
            self.device.close()
            self.device = None

    def read(self):
        """Return the current state of this navigation controller.

        Returns:
            state: {t,x,y,z,pitch,yaw,roll,button} namedtuple
            None if the device is not open.
        """

        try:
            info = self.input_queue.get_all()
            for key, value in info.items():
                self.dict_state[key] = value.tolist()[0]

        except Empty:
            self.dict_state["t"] = high_acc_clock()

        self.tuple_state = SpaceNavigator(**self.dict_state)
        time.sleep(1/self.frequency)

        return self.tuple_state

        
    def run(self):

        print('start')
        self.open()

        while True:
            data = self.device.read(7)

            
            if data[0] == 1:  ## readings from 6-DoF sensor
                self.dict_state["y"] = convert(data[1], data[2])
                self.dict_state["x"] = convert(data[3], data[4])
                self.dict_state["z"] = convert(data[5], data[6]) * -1.0

                self.all_recv[0] = 1

            elif data[0] == 2:

                self.dict_state["roll"] = convert(data[1], data[2])
                self.dict_state["pitch"] = convert(data[3], data[4])
                self.dict_state["yaw"] = convert(data[5], data[6])

                self.all_recv[1] = 1

            elif data[0] == 3:  ## readings from the side buttons
                self.dict_state["buttons"] = np.array([data[1]%2,data[1]//2]) # 1:left,2:right,3:both
                self.all_recv = [1,1]

            # print(high_acc_clock())
            self.dict_state["t"] = high_acc_clock()

            # must receive both parts of the 6DOF state before we return the state dictionary
            if all(num == 1 for num in self.all_recv):
                message = {
                "t": self.dict_state["t"],
                "x": self.dict_state["x"],
                "y": self.dict_state["y"],
                "z": self.dict_state["z"],
                "roll": self.dict_state["roll"],
                "pitch": self.dict_state["pitch"],
                "yaw": self.dict_state["yaw"],
                "buttons": self.dict_state["buttons"],
                }
                self.input_queue.put(message)
                self.all_recv = [0,0]

_active_device = None

def open(vendor_id=9583,product_id=50741,axis_scale=350.0, min_v=-1.0, max_v=1.0,frequency = 60):
    # only used if the module-level functions are used
    global _active_device
    with SharedMemoryManager() as shm_manager:
        new_device = DeviceSpec(shm_manager = shm_manager,vendor_id=vendor_id,product_id=product_id,axis_scale=axis_scale, min_v=min_v, max_v=max_v,frequency=frequency)
        new_device.start()
        _active_device = new_device
        return new_device

def read():
    return _active_device.read()



if __name__ == "__main__":
    dev = open(frequency=60)
    while True:
        r = read()
        print(r.x)