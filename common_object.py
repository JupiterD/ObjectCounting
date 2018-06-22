class CommonObject:
    """
    记录每个物体的信息
    如: 运动轨迹坐标, 有效性
    """

    def __init__(self, rect, point, flag):
        """
        :type rect: list(x1, y1, x2, y2)
        :param rect: 矩形坐标
        :type point: list(x1, y1)
        :param point: 运动轨迹坐标(目标中心)
        :type flag: int
        :param flag: 帧标记. 检测中会向实例发送一个标记, 如果标记与实例标记相同, 则表明在这一帧中存在该物体
        """
        self.rect = list(rect)
        self.points = [point]

        self._frame_flag = flag
        self._has_been_counted = False  # 记录是否已被计数

    def set_counted(self):
        """设置被计数标记, 表明该实例已被计数"""
        self._has_been_counted = True

    def is_counted(self):
        """
        返回该实例是否被计数

        :return: 如果已被计数, 返回True; 否则返回False
        """
        return self._has_been_counted

    def update_point(self, rect, point, flag):
        """
        更新物体运动轨迹坐标

        :type rect: list(x1, y1, x2, y2)
        :param rect: 标记物体的矩形框坐标
        :type point: list(x1, y1)
        :param point: 物体坐标中心 (cx, cy)
        :type flag: int
        :param flag: 帧标记
        """
        if len(self.points) >= 10:
            del self.points[0]

        self.points.append(point)
        self.rect = list(rect)

        self._frame_flag = flag

    def is_exist(self, flag):
        """
        判断帧标记是否与物体标记相等,
        如果相等则表明这一帧中检测到了物体

        :type flag: int
        :param flag: 帧标记
        :return: 存在, 返回True; 不存在, 返回False
        """
        return self._frame_flag == flag

    @staticmethod
    def get_last_point(common_object):
        """
        获取物体实例最后一个运动坐标

        :type common_object: CommonObject
        :param common_object: CommonObject实例
        :return: 坐标
        """
        return common_object.points[-1]

    @staticmethod
    def has_new_flag(object_and_new_flag):
        """
        用于筛选物体, 将不存在的物体删去
        在筛选时应传入一个含有物体实例和帧标记的tuple

        :type object_and_new_flag: list(CommonObject, flag)
        :return: 存在时: True, 不存在时: False
        """
        object_, new_flag = object_and_new_flag
        return object_.is_exist(new_flag)
