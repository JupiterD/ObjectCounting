from common_object import CommonObject
import cv2 as cv


class ObjectCountingPipeline:
    """ 运动物体 检测 - 追踪 - 计数 pipeline """

    def __init__(self, object_detection_model, object_track_model, object_counting_model):
        """
        :param object_detection_model: 物体选择模型
        :param object_track_model: 物体追踪模型
        :param object_counting_model: 物体计数模型
        """
        self._object_detection_model = object_detection_model
        self._object_track_model = object_track_model
        self._object_counting_model = object_counting_model

    def detection_object(self, frame, min_width, min_height, show_mask_frame=False):
        """
        从图像中选择运动物体

        :param frame: 二值化图像
        :param min_width: 物体的最小宽度
        :param min_height: 物体的最小高度
        :param show_mask_frame: 是否显示去噪后的图像
        :return: 每个物体的矩形框左上角x1, y1坐标, 右下角x2, y2坐标和物体中心坐标cx, cy
                [(x1, y1, x2, y2), (cx, cy)]
        """
        frame_mask = self._object_detection_model.background_update(frame)  # 标记图像中的运动物体
        filter_mask = self._object_detection_model.filter_mask(frame_mask)  # 将图像去噪
        matches_list = self._object_detection_model.detect_object(filter_mask, min_width, min_height)  # 计算运动物体的坐标

        if show_mask_frame:
            cv.imshow("img", filter_mask)

        return matches_list

    def object_track(self, object_list, new_matches_list):
        """
        追踪运动目标

        :param object_list: 物体列表
        :param new_matches_list: 检测到的新的物体坐标
        :return: 更新后的物体列表
        """
        new_object_list = self._object_track_model.object_track(object_list, new_matches_list)

        return new_object_list

    def object_counting(self, object_list):
        """
        物体计数

        :param object_list: 物体列表
        :return: 新检测到并符合要求的物体个数
        """
        object_in, object_out, new_counting_list = self._object_counting_model.get_object_count(object_list)

        return object_in, object_out, new_counting_list

    def run(self, frame, min_width, min_height, object_list, counting_log=None, show_mask_frame=False):
        """
        运行计数流程

        :param frame: 图像
        :param min_width: 物体最小宽度
        :param min_height: 物体最小高度
        :param object_list: 物体列表
        :param counting_log: 物体计数日志
        :param show_mask_frame: 是否显示去噪后的图像
        :return: 追踪到的物体列表object_list, 进入的物体个数object_in, 出去的物体个数object_out
        """
        matches_list = self.detection_object(frame, min_width, min_height, show_mask_frame)

        # 如果已存在物体则开始追踪
        if object_list:
            object_list = self.object_track(object_list, matches_list)

        # 否则需添加物体
        else:
            for rect, point in matches_list:
                new_object = CommonObject(rect, point, self._object_track_model.get_frame_flag())
                object_list.append(new_object)

        object_in, object_out, new_counting_list = self.object_counting(object_list)

        if counting_log:
            counting_log.update_counting_pic_list(frame, new_counting_list)

        return object_list, object_in, object_out
