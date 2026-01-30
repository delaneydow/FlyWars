# test_camera.py
from arena_api.system import system

devices = system.create_device()
print(devices)

device = devices[0]
device.start_stream()

for i in range(100):
    buffer = device.get_buffer()
    print(buffer.width, buffer.height)
    device.requeue_buffer(buffer)

