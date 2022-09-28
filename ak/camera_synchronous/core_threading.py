import math
import random
from .utils.utils import default_capture_process
from typing import Callable, List, Union
import pykinect_azure as pykinect
from threading import Thread, Lock
import time
from queue import Empty, Queue
import os


# master默认配置
master_default_configuration = pykinect.Configuration()
master_default_configuration.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_MASTER
master_default_configuration.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
master_default_configuration.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
master_default_configuration.synchronized_images_only = True
master_default_configuration.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
# master_default_configuration.camera_fps = pykinect.k4a._k4a.K4A_FRAMES_PER_SECOND_15

# subordinate默认配置
subordinate_default_configuration = pykinect.Configuration()
subordinate_default_configuration.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_SUBORDINATE
subordinate_default_configuration.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
subordinate_default_configuration.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
subordinate_default_configuration.synchronized_images_only = True
subordinate_default_configuration.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32


def capture_producer(flag, device_id: int, config: pykinect.Configuration, queue: Queue,
                     capture_process: Callable = default_capture_process):
    """capture

    Args:
        flag (_type_): 信号, 2表示退出
        device_id (int): 设备id
        config (pykinect.Configuration): 配置
        queue (Queue): 当前capture队列
    """
    # pykinect.initialize_libraries()
    try:
        device = pykinect.start_device(device_index=device_id, config=config)
        bodytracker = pykinect.start_body_tracker(model_type=0,calibration=device.calibration)
    except Exception as e:
        print(f'start {device_id} failed {e}')
        flag.value = 1
        return
    to_second = 1e6
    offsets = []
    for _ in range(10):
        local_time_before = time.time()
        capture = device.update()
        color_image_object = capture.get_color_image_object()
        ak_time = pykinect.k4a._k4a.k4a_image_get_timestamp_usec(color_image_object.handle()) / to_second
        local_time_after = time.time()
        local_time = (local_time_before + local_time_after) / 2.0
        offsets.append(local_time - ak_time)
    time_offset = sum(offsets) / len(offsets)
    print(f'start {device_id} success')
    while flag.value != 2:
        capture = device.update()
        bodyframe = bodytracker.update(capture=capture)
        color_image_object = capture.get_color_image_object()
        timestamp = pykinect.k4a._k4a.k4a_image_get_timestamp_usec(color_image_object.handle()) / to_second
        timestamp = timestamp + time_offset
        ret, data = capture_process(capture, color_image_object, timestamp, bodyframe)
        if not ret:
            continue
        queue.put(data)
    print(f'close {device_id}')
    device.close()


def synchronous_producer(flag, lock: Lock, queue: Queue, capture_producer_queue_list: List[Queue],
                         max_dist: float = 10 * 1000):
    """同步

    Args:
        flag (_type_): 信号, 1表示退出
        lock (Lock): 锁, 用于控制满了之后更新
        queue (Queue): 同步队列
        capture_producer_queue_list (List[Queue]): capture生产者队列
        max_dist (float, optional): 同步帧阈值(单位微妙). Defaults to 10*1000.
    """
    print(f'start synchronous')
    while flag.value != 1:
        try:
            ret_list = [q.get(timeout=1) for q in capture_producer_queue_list]  # 设置timeout为了在capture_producer异常停止无数据不阻塞
            max_timestamp = max([ret[0] for ret in ret_list])
            for i in range(len(ret_list)):
                best_dist = max_timestamp - ret_list[i][0]
                while max_timestamp > ret_list[i][0] and best_dist > max_dist:  # 暂时不考虑大于max_timestamp更好的情况
                    new_ret = capture_producer_queue_list[i].get(timeout=1)
                    dist = math.fabs(new_ret[0] - max_timestamp)
                    if dist > best_dist:
                        break
                    # 遇到更近的帧，更新
                    best_dist = dist
                    ret_list[i] = new_ret
                if best_dist > max_dist:
                    break
            else:
                if queue.full():
                    lock.acquire()
                    queue.get()
                    queue.put(ret_list)
                    lock.release()
                else:
                    queue.put(ret_list)
        except Empty:
            pass
    print(f'synchronous_producer exit')
    flag.value = 2  # 退出capture producer
    for q in capture_producer_queue_list:  # 避免阻塞
        try:
            q.get(timeout=0.0001)
        except:
            pass


class MulDeviceSynCapture(object):
    def __init__(self, master_device_id: int = 0, subordinate_device_ids: Union[int, List[int]] = 1,
                 master_config: pykinect.Configuration = master_default_configuration,
                 subordinate_config: pykinect.Configuration = subordinate_default_configuration, synmaxsize: int = 10,
                 capmaxsize: int = 10, max_dist: float = 4e-2,
                 capture_process: Callable = default_capture_process) -> None:
        """多相机同步捕获

        Args:
            master_device_id (int, optional): 主相机设备id. Defaults to 0.
            subordinate_device_ids (Union[int, List[int]], optional): 从相机设备id列表. Defaults to 1.
            master_config (pykinect.Configuration, optional): 主相机配置. Defaults to master_default_configuration.
            subordinate_config (pykinect.Configuration, optional): 从相机配置. Defaults to subordinate_default_configuration.
            synmaxsize (int, optional): 同步队列容量. Defaults to 10.
            capmaxsize (int, optional): capture队列容量. Defaults to 10.
            max_dist (float, optional): 同步帧阈值(单位妙). Defaults to 1e-2.
        Examples:
            >>> mc = MulDeviceSynCapture(0, 1)
            >>> ret = mc.get()
            >>> for ret in mc:
            >>>     print(ret)
            >>> mc.close()
            >>>
            >>> with MulDeviceSynCapture(0, 1) as mc:
            >>>     for ret in mc:
            >>>        print(ret)

        """
        if isinstance(subordinate_device_ids, int):
            subordinate_device_ids = [subordinate_device_ids]
        self.capture_process = capture_process
        self.master_device_id = master_device_id
        self.subordinate_device_ids = subordinate_device_ids
        self.master_config = master_config
        self.subordinate_config = subordinate_config
        self.max_dist = max_dist
        pykinect.initialize_libraries(track_body=True)  # track_body=True
        assert pykinect.Device.device_get_installed_count() >= len(
            self.subordinate_device_ids) + 1, "Not enough devices installed"
        self._master_queue = Queue(maxsize=capmaxsize)
        self._subordinate_queue_list = [Queue(capmaxsize) for _ in range(len(subordinate_device_ids))]
        self.synchronous_queue = Queue(synmaxsize)
        self._flag = type('Value', (object,), {'value': 0})  # 用于传递关闭信号, 0: 启动, 1: 关闭同步进程, 2: 关闭capture进程
        self.thread_list = []
        self._lock = Lock()
        self.start()

    def get(self):
        """获取同步帧
        Returns:
            ret (List[Tuple[int, np.ndarray, np.ndarray]]): 同步帧, 其中第一个元素为时间戳(单位微秒), 第二个元素为彩色图像, 第三个元素为深度图像. 参考capture_producer put内容
        """
        self._lock.acquire()
        while True:
            assert not self._flag.value, "Capture is closed"
            try:
                ret = self.synchronous_queue.get(timeout=0.03)
                break
            except Empty:
                pass
        self._lock.release()
        return ret

    def start(self):
        """启动
        """
        self.thread_list = []
        # 同步进程
        thread = Thread(target=synchronous_producer, args=(
        self._flag, self._lock, self.synchronous_queue, [self._master_queue] + self._subordinate_queue_list,
        self.max_dist))
        thread.start()
        self.thread_list.append(thread)

        # 创建master thread
        thread = Thread(target=capture_producer, args=(
        self._flag, self.master_device_id, self.master_config, self._master_queue, self.capture_process))
        thread.start()
        time.sleep(0.1)
        self.thread_list.append(thread)

        # 创建subordinate thread
        for device_id, queue in zip(self.subordinate_device_ids, self._subordinate_queue_list):
            thread = Thread(target=capture_producer,
                            args=(self._flag, device_id, self.subordinate_config, queue, self.capture_process))
            thread.start()
            self.thread_list.append(thread)

        time.sleep(0.1)  # 等待进程启动
        # 查看是否有进程异常
        for thread in self.thread_list:
            if not thread.is_alive():
                self.close()
                break

    def close(self):
        """关闭
        """
        self._flag.value = 1
        for thread in self.thread_list:
            thread.join()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.get()


if __name__ == '__main__':
    with MulDeviceSynCapture(0, 1, max_dist=20000) as capture:
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
