"""
Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from real_env_test import RealEnv_test
from test_mouse_action import Spacemouse 
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from gripper_ctrl import Gripper

# is_2D_motion is to control the robot to move in plane or space
def main(frequency=30):
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            Spacemouse(shm_manager=shm_manager,is_2D_motion = False) as sm, \
            Gripper(shm_manager=shm_manager) as gp, \
            RealEnv_test(
                    output_dir = './test_record4', 
                    robot_ip='192.168.58.2', 
                    # recording resolution
                    obs_image_resolution=(640,480),
                    frequency=frequency,  #因为后面都根据这个采样频率采样，所以采样频率理论上要小于robot的频率和realsense的频率
                    init_joints=False,
                    enable_multi_cam_vis=False,
                    record_raw_video=True,
                    # number of threads per camera view for video recording (H.264)
                    thread_per_video=3,
                    # video recording quality, lower is better (but slower).
                    video_crf=21,
                    shm_manager=shm_manager
                ) as env:
            
                    stop = False
                    iter_idx = 0
                    stage = 0
                    command_latency = 0.01
                    dt = 1/frequency  
                    t_start = time.monotonic()
                    # env.start() #进入启动
                    # env.start_episode(time.time())

                    # rec_start_time = time.time() + 1
                    is_recording = False

                    out = None
                    robot_tcp = np.zeros(6)
                    T = 0
                    R = 0
                    key_counter.clear()
                    
                    while not stop:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + 1) * dt
                        t_sample = t_cycle_end - command_latency
                        t_command_target = t_cycle_end + dt

                        # pump obs
                        if is_recording:
                            obs = env.get_obs() # 这里存入obs
                            robot_tcp = obs['robot_eef_pose'][0].tolist()
                            T = f'X:{round(robot_tcp[0],2)},Y:{round(robot_tcp[1],2)},Z:{round(robot_tcp[2],2)}'
                            R = f'R:{round(robot_tcp[3],2)},P:{round(robot_tcp[4],2)},Y:{round(robot_tcp[5],2)}' 
                        # print(obs['robot_eef_pose'])


                        # handle key presses
                        press_events = key_counter.get_press_events()
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='q'):
                                # Exit program
                                stop = True
                            elif key_stroke == KeyCode(char='c'):
                                # Start recording
                                env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())# 这里robot开始时有两次循环是用来检测是否正常运行，所以开始时间延后
                                key_counter.clear()
                                is_recording = True
                                print('Recording!')
                            elif key_stroke == KeyCode(char='s'):
                                # Stop recording
                                env.end_episode()
                                key_counter.clear()
                                is_recording = False
                                # env.robot_init()
                                print('Stopped.')
                            elif key_stroke == Key.backspace:
                                # Delete the most recent recorded episode
                                if click.confirm('Are you sure to drop an episode?'):
                                    env.drop_episode()
                                    key_counter.clear()
                                    is_recording = False
                                # delete
                        stage = key_counter[Key.space]

                        precise_wait(t_sample)
                        # get teleop command
                        sm_state = sm.get_motion_state_transformed()

                        sm_left_button = sm.is_button_pressed(0)
                        sm_right_button = sm.is_button_pressed(1)

                        if sm_left_button:
                            gp.exec_gripper(1)
                        elif sm_right_button:
                            gp.exec_gripper(0)

                        # print(sm_state)

                        #这里存入action 和 stage
                        env.exec_actions(
                        actions=[sm_state], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        stages=[stage])

                        precise_wait(t_cycle_end)
                        iter_idx += 1

                        out = env.realsense.get(out=out)
                        bgr0 = out[0]['color']
                        bgr1 = out[1]['color']
                        # depth0 = out[0]['depth']
                        # depth1 = out[1]['depth']
                        

                        show_images = np.hstack((bgr0,bgr1))
                        # show_images2 = np.hstack((depth0,depth1))
                        # cv2.imshow('depth_img',show_images2)

                        episode_id = env.replay_buffer.n_episodes
                        text = f'Episode: {episode_id}, Stage: {stage}'
                        if is_recording:
                            text += ', Recording!'
                        cv2.putText(
                            show_images,
                            text,
                            (10,30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            thickness=2,
                            color=(0,0,255))

                        text2 = str("q-exit |c-record |s-save |Backspace-delete")
                        cv2.putText(
                            show_images,
                            text2,
                            (650,30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255))
                        
                        text3 = str(T)
                        text4 = str(R)
                        cv2.putText(
                            show_images,
                            text3,
                            (650,50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(0,0,255))
                        cv2.putText(
                            show_images,
                            text4,
                            (650,70),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(0,0,255))

                        cv2.imshow('color_img',show_images)
                        key = cv2.pollKey()
                        if key == ord('q'):
                            break

                        precise_wait(t_cycle_end)
                        iter_idx +=1

if __name__ == "__main__":
     main(frequency=20) #相机30帧，所以frequency最好小于30