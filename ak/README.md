# 相机同步

## 依赖
### Windows
> https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md

### Ubuntu
> https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md
> 
> https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/406
> 
> Ubuntu20.04只能调用一个相机，无法同时调用两个，应该是环境问题


### pykinext_azure
> Python调用Azure Kinect SDK
> 
> https://github.com/ibaiGorordo/pyKinectAzure

下载当前项目里面有pykinect_azure, 有一点改动


## 安装
`pip install .`  安装到当前环境中


## 使用
```python
from camera_synchronous import MulDeviceSynCapture
mc = MulDeviceSynCapture(0, 1)
ret = mc.get()
for ret in mc:
    print(ret)
mc.close()


with MulDeviceSynCapture(0, 1) as mc:
    for ret in mc:
        print(ret)
```

## Example
```python
import time
from camera_synchronous import MulDeviceSynCapture
import pykinect_azure as pykinect


def default_capture_process(capture: pykinect.Capture, color_image_object: pykinect.Image, timestamp_usec: int):
    """默认的捕获函数

    Args:
        capture (pykinect.Capture): Capture对象
        color_image_object (pykinect.Image): Image对象
        timestamp_usec (int): 时间戳, 单位微秒

    Returns:
        Tuple[bool, Tuple[Any]]: 第一个返回值表示是否成功, 第二个返回值表示成功的数据
    """
    color_ret, color_image = color_image_object.to_numpy()
    depth_ret, depth_image = capture.get_depth_image()
    return color_ret and depth_ret, (timestamp_usec, color_image, depth_image)


if __name__ == '__main__':
    with MulDeviceSynCapture(0, 1, max_dist=20000, capture_process=default_capture_process) as capture:
        # warm-up
        for i in range(10):
            capture.get()
        start_time = time.time()
        num = 1000
        # while True:
        for i in range(num):
            ret = capture.get()
        fps = num / (time.time() - start_time)
        print(f'synchronous fps: {fps}')
    
```