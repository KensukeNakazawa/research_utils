"""
データ拡張のためのモジュール
"""
# coding: utf-8

import cv2
import numpy as np


class DataAugmentation:

    # def __init__(self):

    @staticmethod
    def rotation(X, y):
        """
        データの水増しを行う

        Args:
            image(ndarray): 水増しを行いたい画像データ
            X(ndarray): 学習データのnumpy配列
            y(ndarray): ラベルデータのnumpy配列
            label(integer): ラベルデータ

        Reterns:
            X(ndarray): 画像データのnumpy配列
            y(ndarray): ラベルデータのnumpy配列
        """
        augmented_X = []
        augmented_y = []
        for label, image in enumerate(X):
            height, width, _ = image.shape[:3]
            center_height = height // 2
            center_width = width // 2
            center = (center_width, center_height)

            for i in range(1, 37):
                trans = cv2.getRotationMatrix2D(center, i*10.0, 1.0)
                image2 = cv2.warpAffine(image, trans, (width, height), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
                augmented_X.append(image2)
                augmented_y.append(y[label])
        augmented_X = np.array(augmented_X)
        augmented_y = np.array(augmented_y)

        return augmented_X, augmented_y
