# 没有摄像头和环境时测试使用
import time
import numpy as np
import mainThread.utils.camera_synchronous.pykinect_azure as pykinect
import random
from PIL import Image, ImageDraw

def initialize_libraries():
    print('[Fake] initialize_libraries')


pykinect.initialize_libraries = initialize_libraries

pykinect.k4a._k4a.k4a_image_get_timestamp_usec = lambda x: x.timestamp_usec

class ImageObject(object):
    def __init__(self, image, timestamp_usec) -> None:
        self.timestamp_usec = timestamp_usec
        self.color_image = image
    
    def to_numpy(self):
        return True, self.color_image
    
    def handle(self):
        return self
    

class Capture(object):
    def __init__(self, image) -> None:
        self.image = image
        
    def get_color_image_object(self):
        return ImageObject(self.image, int(time.time() * 1000 * 1000))
    
    def get_depth_image(self):
        # return True, np.random.randint(500, 10000, (1080, 720))
        return True, np.empty((1080, 720))

class Device(object):
    def __init__(self, device_index: int = 0, config: pykinect.Configuration = pykinect.default_configuration) -> None:
        self.device_index = device_index
        self.config = config
        self.fps = {0: 5, 1: 15, 2: 30}[self.config.camera_fps]
        
    def update(self):
        time.sleep(1 / self.fps + random.random() * (0.1 / self.fps) - (0.05 / self.fps))  # fps波动
        image = Image.new('RGB', (320, 240))
        draw = ImageDraw.Draw(image)
        draw.text((image.width // 2 - 40, image.height // 2 - 10), str(int(time.time() * 1000)), fill='white')
        return Capture(np.array(image))

    def close(self):
        pass

    @staticmethod
    def device_get_installed_count():
        return 999

pykinect.Device = Device

def start_device(device_index: int = 0, config: pykinect.Configuration = pykinect.default_configuration) -> Device:
    if random.random() < 0.05:
        raise Exception(f'[Fake] Start K4A cameras failed!')
    print(f'[Fake] start_device {device_index}')
    return Device(device_index, config)

pykinect.start_device = start_device
