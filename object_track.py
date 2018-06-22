from common_object import CommonObject

import numpy as np


class ObjectTrackKNN:
    """ 追踪多个运动物体 """

    def __init__(self, split_line, centroid_threshold_square):
        """
        :type split_line: int
        :param split_line: 将图像分割为两部分的线(取下半部分为识别区域)
        :type centroid_threshold_square: int
        :param centroid_threshold_square: 运动轨迹最大距离的平方
        """
        self._split_line = split_line
        self.centroid_threshold_square = centroid_threshold_square
        self._frame_flag = 0

    def _update_frame_flag(self):
        """更新帧标记"""
        self._frame_flag = (self._frame_flag + 1) % 10000

    def _calculate_distance_square(self, object_last_point_list, x, y):
        """
        计算新点与每个旧点的距离

        :param object_last_point_list: 所有物体上一次运动到的坐标集合
        :param x: 新的点的x坐标
        :param y: 新的点的y坐标
        :return: 距离的平方
        """
        cx_square = np.square(object_last_point_list[:, 0] - x)
        cy_square = np.square(object_last_point_list[:, 1] - y - self._split_line)
        distance_square = cx_square + cy_square

        return distance_square

    def _calculate_offset(self, y1, y2, new_cy):
        """
        计算y轴偏移坐标

        将y轴上的坐标加上偏移量, 因为在检测时只有整个图像的split_line线下的部分
        因此要加上split_line的偏移量, 才能与原图像的区域坐标对应

        :param y1: 矩形y1坐标
        :param y2: 矩形y2坐标
        :param new_cy: 矩形中心y坐标
        :return: 偏移坐标
        """
        y1 += self._split_line
        y2 += self._split_line
        new_cy += self._split_line

        return y1, y2, new_cy

    def _update_object_list(self, object_list):
        """
        删除未被新的帧标记标记过的物体对象

        :param object_list: 物体列表
        :return: 更新后的物体列表
        """
        # 清除未被标记的物体对象
        temp = filter(CommonObject.has_new_flag, zip(object_list, [self._frame_flag] * len(object_list)))
        object_list = [each[0] for each in temp]  # temp中的每个元素的格式为(object, flag), 只需获取object

        return object_list

    def get_frame_flag(self):
        """获取当前帧标记"""
        return self._frame_flag

    def object_track(self, object_list, new_matches_list):
        """
        计算物体运动轨迹, 将新的轨迹坐标添加到对应的object中, 并更新object_list列表

        :param object_list: 物体列表
        :param new_matches_list: 检测到的新的物体坐标
        :return: 物体列表
        """
        self._update_frame_flag()

        get_last_points = map(CommonObject.get_last_point, object_list)  # 获取每个物体上一次所在位置的坐标
        object_last_point_list = np.array([last_point for last_point in get_last_points])

        # 利用KNN算法的思想, 将新的坐标进行分类, 匹配最近的轨迹
        for (x1, y1, x2, y2), (new_cx, new_cy) in new_matches_list:
            # 逼近分割线, 避免识别到的矩形框变形严重(过小)
            if new_cy < 30:
                continue

            distance_square = self._calculate_distance_square(object_last_point_list, new_cx, new_cy)

            y1, y2, new_cy = self._calculate_offset(y1, y2, new_cy)

            # 判断该点是否属于已存在的物体
            is_exist = distance_square.min() < self.centroid_threshold_square
            if is_exist:
                min_distance_index = distance_square.argmin()  # 取出与新点距离平方最小的(匹配的)旧点的序号
                selected_object = object_list[min_distance_index]  # 根据匹配的旧点序号, 取出对应的物体

                selected_object.update_point((x1, y1, x2, y2), (new_cx, new_cy), self._frame_flag)  # 更新物体运动轨迹

            # 该点不属于任何已存在的物体
            else:
                # 创建新的物体并添加到object_list中
                new_object = CommonObject((x1, y1, x2, y2), (new_cx, new_cy), self._frame_flag)
                object_list.append(new_object)

        object_list = self._update_object_list(object_list)  # 去除未被标记的物体

        return object_list
