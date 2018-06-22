from object_detection import ObjectDetection
from object_track import ObjectTrackKNN
from object_counting import ObjectCounting
from object_counting_log import ObjectCountingLog
from object_counting_pipeline import ObjectCountingPipeline

import cv2 as cv
import numpy as np
from math import sqrt


def draw_frame(frame, object_list, string_and_coordinate, font, counting_log=None, split_line=None, counting_line=None):
    """
    绘制图像

    :param frame: 图像
    :type object_list: list
    :param object_list: 物体列表
    :type string_and_coordinate: list
    :param string_and_coordinate: 显示的字符串和坐标的列表
    :param font: 字体
    :param counting_log: 物体计数日志
    :type split_line: int
    :param split_line: 检测分割线的y轴坐标
    :type counting_line: int
    :param counting_line: 计数线的y轴坐标
    :return: frame
    """
    height, width = frame.shape[:2]
    show_frame = np.zeros((height + 160, width, 3), dtype=np.uint8)
    show_frame[:height, :] = frame

    # 绘制分割线
    if split_line:
        cv.line(show_frame, (0, split_line), (1280, split_line), (0, 255, 0), 2)

    # 绘制计数线
    if counting_line:
        cv.line(show_frame, (0, counting_line), (1280, counting_line), (255, 255, 0), 1)

    # 绘制物体追踪矩形和运动轨迹
    for each_object in object_list:
        # 绘制矩形
        x1, y1, x2, y2 = each_object.rect
        cv.rectangle(show_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        centroid_count = len(each_object.points)
        color = (0, 0, 255)
        # 存在两个以上的点, 可以绘制轨迹
        if centroid_count >= 2:
            # 被计数过的物体和未被计数过的的物体轨迹颜色不同
            if each_object.is_counted():
                color = (0, 255, 255)

            for i in range(centroid_count - 1):
                thickness = int(sqrt((i + 1) * 2.5))  # 运动轨迹线条粗细
                cv.line(show_frame, each_object.points[i], each_object.points[i + 1], color, thickness)

        # 只存在一个点, 标记中心
        else:
            cv.circle(show_frame, each_object.points[0], 1, color, 1)

    if counting_log:
        for i, object_pic in enumerate(counting_log.get_counting_pic_list()):
            show_frame[height:, (7 - i) * 160:(8 - i) * 160] = object_pic

    for string, coordinate in string_and_coordinate:
        cv.putText(show_frame, string, coordinate, font, 1, (0, 0, 0), 1)

    return show_frame


if __name__ == '__main__':
    print("打开视频")
    video = "./video/3.mp4"
    cap = cv.VideoCapture(video)

    print("初始化物体选择模型")
    history = 500
    var_threshold = 64
    learn_rate = 0.005
    bg_subtractor = cv.createBackgroundSubtractorMOG2(history, var_threshold, detectShadows=False)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    detection_model = ObjectDetection(bg_subtractor, history, learn_rate, kernel)
    detection_model.train_model(cap)

    print("初始化物体追踪模型")
    split_line = 368
    centroid_threshold_square = 1300
    track_model = ObjectTrackKNN(split_line, centroid_threshold_square)

    print("初始化物体计数模型")
    counting_line = split_line + 50
    counting_model = ObjectCounting(counting_line)

    print("初始化物体计数日志")
    counting_log = ObjectCountingLog(split_line)

    print("初始化pipeline")
    pipeline = ObjectCountingPipeline(detection_model, track_model, counting_model)

    print("初始化视频输出器")
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output_{}_{}_{}.avi'.format(history, var_threshold, learn_rate), fourcc, 25.0, (1280, 880))

    font = cv.FONT_HERSHEY_SIMPLEX

    object_list = []  # 检测到的物体列表
    counting_pic_list = []  # 记录已计数的物体
    object_in = object_out = 0  # 物体的进出个数
    start_time = end_time = total_time = fps = 0  # 计算fps所需
    tick_frequency = cv.getTickFrequency()

    fps_string = "fps: 0"

    retval, frame = cap.read()
    while retval:
        frame_temp = frame[split_line:, :]

        start_time = cv.getTickCount()
        object_list, new_object_in, new_object_out = pipeline.run(frame_temp, 35, 35, object_list, counting_log)
        object_in += new_object_in
        object_out += new_object_out
        end_time = cv.getTickCount()

        counting_string = "in: {}  out: {}".format(object_in, object_out)

        string_and_coordinate = [(counting_string, (40, 40)), (fps_string, (1100, 40))]

        frame = draw_frame(frame, object_list, string_and_coordinate, font, counting_log, split_line, counting_line)

        cv.imshow("video", frame)

        out.write(frame)

        key = cv.waitKey(10) & 0xff

        retval, frame = cap.read()

        fps += 1

        # 一秒更新一次fps
        total_time += (end_time - start_time) / tick_frequency
        if total_time >= 1:
            fps_string = "fps: {}".format(fps)
            fps = 0
            total_time = 0

        if key == ord('q'):
            break
        elif key == ord(' '):
            cv.waitKey(0)

    cv.destroyAllWindows()
    cap.release()
