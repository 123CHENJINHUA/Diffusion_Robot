import sys
sys.path.insert(0,sys.path[0]+'/../')
import time
import Robot
import numpy as np
import enum
import multiprocessing as mp
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from multiprocessing.managers import SharedMemoryManager


class recv_list(object):
    def __init__(self,robot_r):
        self.ActualTCPPose = robot_r.GetActualTCPPose()[1]
        self.ActualTCPSpeed = robot_r.GetActualTCPSpeed()[1]
        self.ActualQ = robot_r.GetActualJointPosDegree()[1]
        self.ActualQd = robot_r.GetActualJointSpeedsDegree()[1]


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    ROBOT_INIT = 3      

class Robot_control(mp.Process):
    def __init__(self,shm_manager: SharedMemoryManager, 
            robot_ip, 
            frequency=125, 
            receive_keys=None,
            get_max_k=128,
            launch_timeout=3,
            verbose=False,
            robot_latency = 0.001):
        super().__init__()
        # self.actualpose = np.zeros(6)
        self.robot_ip = robot_ip
        self.launch_timeout = launch_timeout
        self.verbose = verbose

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=1024#256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd'
            ]

        rtde_r = Robot.RPC(self.robot_ip)
        robot_list = recv_list(rtde_r)
        example = dict()
        for key in receive_keys:
            example[key] = np.array(getattr(robot_list, key))
        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
        self.frequency = frequency
        self.is_init = False
        self.robot_latency = robot_latency

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def robot_setup(self):
        #------------Setup connecting to robot arm----------#
        # 与机器人控制器建立连接，连接成功返回一个机器人对象
        print("connected")
        self.rtde_c = Robot.RPC(self.robot_ip)
        self.rtde_r = Robot.RPC(self.robot_ip)

    def recv_data(self):
        # self.actualpose = self.rtde_r.GetActualTCPPose()
        # self.actualpose = self.actualpose[1]

        robot_list = recv_list(self.rtde_r)
        # update robot state
        state = dict()
        
        for key in self.receive_keys:
            state[key] = np.array(getattr(robot_list, key))  # 这里获取数值
        state['robot_receive_timestamp'] = time.time()
        self.ring_buffer.put(state)

    def print_recv(self):
        state = self.ring_buffer.get()
        P = state['ActualTCPPose']
        J = state['ActualQ']
        SP = state['ActualTCPSpeed']
        SJ = state['ActualQd']

        print('TCPPose: ------------> X:%f,Y:%f,Z:%f,R:%f,P:%f,Y:%f' %(round(P[0],2),round(P[1],2),round(P[2],2),round(P[3],2),round(P[4],2),round(P[5],2)))
        print('JointPose: ------------> J1:%f,J2:%f,J3:%f,J4:%f,J5:%f,J6:%f' %(round(J[0],2),round(J[1],2),round(J[2],2),round(J[3],2),round(J[4],2),round(J[5],2)))
        print('TCPSpeed: ------------> X:%f,Y:%f,Z:%f,R:%f,P:%f,Y:%f' %(round(SP[0],2),round(SP[1],2),round(SP[2],2),round(SP[3],2),round(SP[4],2),round(SP[5],2)))
        print('JointSpeed: ------------> J1:%f,J2:%f,J3:%f,J4:%f,J5:%f,J6:%f' %(round(SJ[0],2),round(SJ[1],2),round(SJ[2],2),round(SJ[3],2),round(SJ[4],2),round(SJ[5],2)))


        time.sleep(1/self.frequency)

# ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)
        
    def robot_moving(self,pose,vel=5):
        mode = 2  #[0]-绝对运动(基坐标系)，[1]-增量运动(基坐标系)，[2]-增量运动(工具坐标系)
        # error = self.robot.ServoMoveStart()
        error = self.rtde_c.ServoCart(mode, pose, vel=vel,cmdT = 0.01)   #笛卡尔空间伺服模式运动
        if error != 0:
            print("error:",error)

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()


    def robot_init(self):
        message = {
            'cmd': Command.ROBOT_INIT.value,
        }
        self.input_queue.put(message)

    def run(self):
        self.robot_setup()
        iter_idx = 0
        t_now = time.monotonic()
        dt = 1. / self.frequency

        keep_running = True
        target_pose = np.zeros(6)

        error_appear = False
        
        

        while keep_running:

            #robot initial
            if not self.is_init:
                print('robot_init-------->>>>')
                tool = 0
                user = 0
                init_pose = [-539.277,-100.277,307.539,179.723,0.826,6.636]
                error = self.rtde_c.MoveL(init_pose, tool, user, vel=10, acc=100)   #笛卡尔空间伺服模式运动
                if error != 0:
                    print("error:",error)
                print('robot ready!!!!!')
                self.is_init = True

            if iter_idx == 0:
                self.ready_event.set()
            iter_idx = 1
            

            # print(target_pose)
            if error_appear:
                try: 
                    self.rtde_c = Robot.RPC(self.robot_ip)
                    self.rtde_r = Robot.RPC(self.robot_ip)
                except:
                    print("time out again")
                    continue
                finally:
                    print("Fix!")
                    error_appear = False

            try:
                self.recv_data()
            except:
                time.sleep(0.008)
                error_appear = True
                print('recv error appear!')
                continue

            try:
                self.robot_moving(target_pose,vel=5)   #笛卡尔空间伺服模式运动
            except:
                time.sleep(0.008)
                error_appear = True
                print('ctl error appear!')
                continue
            

        # fetch command from queue
            try:
                commands = self.input_queue.get_all()
                n_cmd = len(commands['cmd'])
            except Empty:
                n_cmd = 0

            # execute commands
            for i in range(n_cmd):
                command = dict()
                for key, value in commands.items():
                    command[key] = value[i]
                cmd = command['cmd']

                if cmd == Command.STOP.value:
                    keep_running = False
                    # stop immediately, ignore later commands
                    break
                elif cmd == Command.SERVOL.value:
                    # since curr_pose always lag behind curr_target_pose
                    # if we start the next interpolation with curr_pose
                    # the command robot receive will have discontinouity 
                    # and cause jittery robot behavior.
                    target_pose = command['target_pose']
                    duration = float(command['duration'])
                    curr_time = t_now + dt
                    t_insert = curr_time + duration
                    
                    if self.verbose:
                        print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                            target_pose, duration))
                elif cmd == Command.SCHEDULE_WAYPOINT.value:
                    target_pose = command['target_pose']
                    target_time = float(command['target_time'])
                    # translate global time to monotonic time
                    target_time = time.monotonic() - time.time() + target_time
                    curr_time = t_now + dt
                elif cmd == Command.ROBOT_INIT.value:
                    self.is_init = False
                    
                else:
                    keep_running = False
                    break

                

            time.sleep(self.robot_latency)

if __name__ == "__main__":
    with SharedMemoryManager() as shm_manager:
        calss_robot = Robot_control(shm_manager = shm_manager, robot_ip='192.168.58.2') #self.shm_manager = shm_manager 这个的问题，删了就行
        
        calss_robot.start()  # 只有用with as 才默认调用__enter__

        assert calss_robot.is_ready

        while True:
            calss_robot.print_recv()
            
