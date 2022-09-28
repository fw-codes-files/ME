import time
from camera_synchronous import MulDeviceSynCapture


def test_fps():
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
        assert 25 < fps < 35, "fps: {}".format(fps)

def test_exception():
    try:
        with MulDeviceSynCapture(0, 1, max_dist=20000) as capture:
            raise Exception("exception test")
    except:
        pass
    else:
        raise Exception("exception caught")
            