import cv2 as cv


class ObjectCountingLog:
    """ 物体计数日志 """

    def __init__(self, split_line):
        self.split_line = split_line
        self._counting_pic_list = []

    def update_counting_pic_list(self, frame, new_counting_list):
        """
        更新物体图像列表

        :param frame: 图像
        :param new_counting_list: 被新追踪到的物体列表
        """
        # 维护此列表最多只有8个物体
        if len(new_counting_list) + len(self._counting_pic_list) > 8:
            del self._counting_pic_list[:len(new_counting_list)]

        for each_object in new_counting_list:
            x1, y1, x2, y2 = each_object.rect
            object_pic = frame[y1 - self.split_line:y2 - self.split_line, x1:x2]
            object_pic = cv.resize(object_pic, (160, 160))

            self._counting_pic_list.append(object_pic)

    def get_counting_pic_list(self):
        """获取图像列表"""
        return self._counting_pic_list
