import spacenavigator
import time

success = spacenavigator.open()
if success:
  while 1:
    state = spacenavigator.read()
    print('xyz------------')
    print(state.x, state.y, state.z)
    print('rpy------------')
    print(state.roll, state.pitch, state.yaw)
    print('buttons----------')
    print(state.buttons)


    time.sleep(0.5)

spacenavigator.close()