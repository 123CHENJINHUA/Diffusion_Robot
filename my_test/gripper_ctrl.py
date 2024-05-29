import serial
import sys
sys.path.insert(0,sys.path[0]+'/../')
import multiprocessing as mp
import numpy as np
import time
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from multiprocessing.managers import SharedMemoryManager


class Gripper(mp.Process):
    def __init__(self,
            shm_manager,
            get_max_k=30, 
            serialPort = 'com4',
            baudRate = 9600,
            frequency=200,
            ):
        super().__init__()

        example = {
            'gripper_state': np.zeros(1, dtype=bool),
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        
        self.frequency = frequency
        self.is_grasp = False
        self.serialPort = serialPort
        self.baudRate = baudRate 
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

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

    def Setup(self,serialPort,baudRate):
        ser = serial.Serial(serialPort,baudRate,timeout=0.5)
        print('SerialPort = %s, BaudRate = %d' % (serialPort,baudRate))
        return ser


    def exec_gripper(self,is_grasp):

        if is_grasp:
            gripper_state = True

        else:
            gripper_state = False

        # send one message immediately so client can start reading
        self.ring_buffer.put({
            'gripper_state': gripper_state,
            'receive_timestamp': time.time()
        })

    def get_gripper_state(self):
        state = self.ring_buffer.get()
        return state['gripper_state']

    def run(self):
        ser = self.Setup(serialPort = self.serialPort, baudRate=self.baudRate)
        print("gripper start!")
        send_word_stop = b'0'
        send_word_start = b'1'
        try:
            gripper_state = np.zeros(1, dtype=bool)
            # send one message immediately so client can start reading
            self.ring_buffer.put({
                'gripper_state': gripper_state,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()

            while not self.stop_event.is_set():

                gripper_state = self.get_gripper_state()
                # print(gripper_state)
                if gripper_state:
                    while 1:
                        ser.write(send_word_start)
                        str = ser.readline()
                        # print(str)
                        if str == b'1': break
                else:
                    while 1:
                        ser.write(send_word_stop)
                        str = ser.readline()
                        # print(str)
                        if str == b'0': break

                time.sleep(1/self.frequency)

        finally:
            while 1:
                ser.write(send_word_stop)
                str = ser.readline()
                # print(str)
                if str == b'0': break
            ser.close()
        

if __name__ == "__main__":
    with SharedMemoryManager() as shm_manager:
        gp = Gripper(shm_manager=shm_manager)
        gp.start()

        gp.exec_gripper(1)
        while True:
            # state = Sp.ring_buffer.get()
            # print(state['motion_event'])

            # sm_state = sm.get_motion_state_transformed()
            
            gp_state = gp.get_gripper_state()
            print(gp_state)
