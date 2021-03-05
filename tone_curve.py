# coding: utf-8

import cv2
import numpy as np


class ToneCurve:

    def __init__(self):
        print('Success Defined!')

    @staticmethod
    def s(target_images):
        """

            Args:
                target_images: 変換したい対象RGB画像

            Returns:
                filtered_images: S字トーンカーブにより変換された画像
            """

        look_up_table = np.zeros((256, 1), dtype='uint8')

        for i in range(256):
            look_up_table[i][0] = 255 * (np.sin((i / 255 - 1 / 2) * np.pi) + 1) / 2

        filtered_images = []
        for target_image in target_images:
            filtered_image = cv2.LUT(target_image, look_up_table)
            filtered_images.append(filtered_image)

        filtered_images = np.array(filtered_images)

        return filtered_images
