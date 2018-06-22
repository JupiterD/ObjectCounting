import cv2 as cv
import numpy as np
from imutils.object_detection import non_max_suppression


class ObjectDetection:
    """ 检测并选择运动物体 """

    def __init__(self, model, model_history, learn_rate, kernel):
        """
        :param model: 物体检测模型
        :param model_history: 物体检测模型训练次数
        :param learn_rate: 模型学习速率
        :param kernel: 去噪时所用的核
        """
        self._model = model  # 物体检测模型
        self._model_history = model_history  # 物体检测模型训练次数
        self.learn_rate = learn_rate  # 算法模型的学习速率

        self._kernel = kernel  # 去噪时所用的矩阵

    def train_model(self, cap):
        """
        训练模型

        :param cap: 视频流
        """
        print("开始训练模型\n模型: {}\n训练次数: {}".format(self._model, self._model_history))
        for i in range(self._model_history):
            retval, frame = cap.read()

            if retval:
                self._model.apply(frame, None, self.learn_rate)
            else:
                raise IOError("图像获取失败")

    def background_update(self, frame):
        """
        利用模型更新背景

        :param frame: 新的帧
        :return: 更新后的图像
        """
        return self._model.apply(frame, None, self.learn_rate)

    def filter_mask(self, frame):
        """
        将图像去噪

        :param frame: 视频帧
        :return: 去噪后的二值化图像
        """
        closing = cv.morphologyEx(frame, cv.MORPH_CLOSE, self._kernel)
        opening = cv.morphologyEx(closing, cv.MORPH_OPEN, self._kernel)

        expend = cv.dilate(opening, self._kernel, iterations=2)
        erode = cv.erode(expend, self._kernel)

        # 清除低于阀值噪点, 因为可能还存在灰色像素
        threshold = cv.threshold(erode, 240, 255, cv.THRESH_BINARY)[1]

        return threshold

    def detect_object(self, frame, min_width=35, min_height=35):
        """
        将二值化图像中的物体挑选出来

        :param frame: 二值化图像
        :param min_width: 物体最小宽度
        :param min_height: 物体最小高度
        :return: 每个物体的矩形框左上角x1, y1坐标, 右下角x2, y2坐标和物体中心坐标cx, cy
                [(x1, y1, x2, y2), (cx, cy)]
        """
        matches = []

        # 找到物体边界矩形
        image, contours, hierarchy = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)

        # 利用非极大值抑制法避免一个物体上多个矩形(误检测多次)
        rects = np.array([(x, y, x + w, y + h) for x, y, w, h in map(cv.boundingRect, contours)])
        pick = non_max_suppression(rects, overlapThresh=0.65)

        # 从每个坐标中选出符合标准大小的坐标(物体)
        for x1, y1, x2, y2 in pick:
            # 判断物体大小是否大于设定的标准
            is_valid = (x2 - x1 > min_width) and (y2 - y1 > min_height)

            # 符合标准, 将矩形坐标和物体中心坐标添加到列表中
            if is_valid:
                centroid = self._get_centroid(x1, y1, x2, y2)

                matches.append([(x1, y1, x2, y2), centroid])

        return matches

    @staticmethod
    def _get_centroid(x1, y1, x2, y2):
        """
        获取物体中心

        :param x1: x轴起点
        :param y1: y轴起点
        :param x2: x轴终点
        :param y2: y轴终点
        :return: 中心坐标 (cx, xy)
        """
        return ((x2 + x1) // 2), ((y2 + y1) // 2)
