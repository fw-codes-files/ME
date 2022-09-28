from camera_synchronous.core_threading import MulDeviceSynCapture
import cv2
import pykinect_azure as pykinect

def imshow(v:list):
    print(v[0],v[1],v[9])
class PackageOfAK(object):
    def __init__(self):
        pass

    @classmethod
    def autoRunAK(cls):
        ak_num = 0
        for i in range(20):
            detector = cv2.VideoCapture(i)
            ret, frame = detector.read()
            if ret:
                continue
            else:
                ak_num = i
                break
        if ak_num == 0:
            print('There is no AK detected on this machine! Please check again!')
            return 0
        elif ak_num == 1:
            pykinect.initialize_libraries()
            device_config = pykinect.default_configuration
            device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
            device = pykinect.start_device(config=device_config)
            while True:
                capture = device.update()
                ret, color_image = capture.get_color_image()
                if not ret:
                    continue
                cv2.imshow("Color Image", color_image)
                if cv2.waitKey(1) == ord('q'):
                    return
        else:
            master = 0
            sub_list = [sub for sub in range(1, ak_num)]
            mc = MulDeviceSynCapture(master,sub_list)
            ret = mc.get()
            variables = locals()
            while 1:
                for ak in range(len(ret)):
                    #
                    variables['img%s' % ak] = ret[ak][1]
                    cv2.imshow(f'img{ak}',variables['img%s' % ak])
                    cv2.waitKey(1)

            mc.close()
if __name__ == '__main__':
    v = locals()
    l = list
    for j in range(4):
        for i in range(10):
            v['name%s'%i] = i+j
    print(v.keys())