"""
画像読み込みのためのモジュール
InputImage V2
"""
# coding: utf-8
"""
"""

import glob
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


class InputImageV2:
    """
    画像の読み込みを行うクラス
    """

    def __init__(self, image_size=(28, 28), train_size=0.8):
        """
        クラスの初期化を行う

        Args:
            classes(list): 学習したい画像のラベルリスト
            image_size(tuple):   画像をリサイズするサイズ
            max_read_images(int):  画像を読み込む最大値
        """
        self.classes = [1, 2, 3, 4, 5, 6]
        self.image_size = image_size
        self.image_extension = '.bmp'
        self.max_read_images = 500
        self.train_size = train_size
        self.test_size = 1 - self.train_size


    def get_test_train_data(self):
        """
        画像データを読み込み、学習データとテストデータに分けて返す

        Returns:
            X_train(ndarray): 学習データ(image)
            X_test(ndarray): テストデータ(image))
            y_train(ndarray): 学習データ(label)
            y_test(ndarray): テストデータ(label)
        """
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=self.train_size,
            test_size=self.test_size,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def load_data(self):
        """
        画像を読み込んで，教師データとラベルを返す

        Reterns:
        X(ndarray): 画像データのnumpy配列
        y(ndarray): ラベルデータのnumpy配列
        """
        image_path = os.pardir + '/images/'
        X = []
        y = []
        for index, class_label in enumerate(self.classes):
            images_dir = image_path + str(class_label)
            files = glob.glob(images_dir + "/*{}".format(self.image_extension))
            for j_index, file in enumerate(files):
                if j_index == self.max_read_images:
                    break
                image = cv2.imread(file, cv2.IMREAD_COLOR)
                image = cv2.resize(image, self.image_size)
                X.append(image)
                y.append(index)

        X = np.array(X)
        y = np.array(y)

        return X, y
