import pykinect_azure as pykinect



def default_capture_process(capture: pykinect.Capture, color_image_object: pykinect.Image, timestamp_usec: int,bodyframe):
    """默认的捕获函数

    Args:
        capture (pykinect.Capture): Capture对象
        color_image_object (pykinect.Image): Image对象
        timestamp_usec (int): 时间戳, 单位微秒

    Returns:
        Tuple[bool, Tuple[Any]]: 第一个返回值表示是否成功, 第二个返回值表示成功的数据
    """
    color_ret, color_image = color_image_object.to_numpy()
    color_skeleton, bn = bodyframe.draw_bodies(color_image.copy(), pykinect.K4A_CALIBRATION_TYPE_COLOR)
    if bn == 0:
        return color_ret, (timestamp_usec,color_image,bn,0,color_skeleton)
    else:
        joints = bodyframe.get_body()
        return color_ret, (timestamp_usec,color_image,bn,joints,color_skeleton)