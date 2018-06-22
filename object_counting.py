class ObjectCounting:
    """ 物体计数 """

    def __init__(self, counting_line):
        """
        :type counting_line: int
        :param counting_line: 计数线的y轴坐标
        """
        self.counting_line = counting_line

    def get_object_count(self, object_list):
        """
        获取物体个数

        :param object_list: 物体列表
        :return: 物体进出个数和被计数的物体列表
        """
        object_in = object_out = 0
        new_counting_list = []

        for each_object in object_list:
            # 运动轨迹至少为两个, 否则无法判断
            # 并且该物体还未被计数才可计数
            if not each_object.is_counted() and len(each_object.points) > 1:
                # 需要计算运动轨迹中最后两个点的向量来判断运动方向
                new_cx, new_cy = each_object.points[-1]
                last_cx, last_cy = each_object.points[-2]

                # 判断两点是否在计数线两侧, 如果在两侧则为负数(等于零也算在两侧), 否则为正数
                is_valid = (self.counting_line - new_cy) * (self.counting_line - last_cy) <= 0
                if is_valid:
                    # 如果出去则最新的点在计数线上方
                    if new_cy - last_cy <= 0:
                        object_out += 1
                    else:
                        object_in += 1

                    new_counting_list.append(each_object)
                    each_object.set_counted()

        return object_in, object_out, new_counting_list
