"""
画像読み込みのためのモジュール
"""
# coding: utf-8

import glob
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


class InputImage:
    """
    画像の読み込みを行うクラス
    """

    def __init__(self, image_size=(28, 28), train_size=0.75):
        """
        クラスの初期化を行う

        Args:
            classes(list): 学習したい画像のラベルリスト
            image_size(tuple):   画像をリサイズするサイズ
            max_read_images(int):  画像を読み込む最大値
        """
        self.classes = [1, 2, 3, 4]
        self.image_size = image_size
        self.max_read_images = 25
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
            test_size=self.test_size
        )
        X_train, y_train = self.__data_augment_v2(X_train, y_train)

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
          files = glob.glob(images_dir + '/*.png')
          for _, file in enumerate(files):
            image = cv2.imread(file, cv2.IMREAD_COLOR)
            image = cv2.resize(image, self.image_size)
            X.append(image)
            y.append(index)
          # X, y = self.__data_augment(image, X, y, index)

        X = np.array(X)
        y = np.array(y)

        return X, y

    def __data_augment(self, image, X, y, label):
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
        height, width, _ = image.shape[:3]
        center_height = height // 2
        center_width = width // 2
        center = (center_width, center_height)

        for i in range(0, 37):
             trans = cv2.getRotationMatrix2D(center, i*10.0, 1.0)
             image2 = cv2.warpAffine(image, trans, (width, height))
             X.append(image2)
             y.append(label)

        return X, y

    def __data_augment_v2(self, X, y):
        """
        データの水増しを行う

        Args:
            X(ndarray): 水増しを行いたい学習データのnumpy配列
            y(ndarray): ラベルデータのnumpy配列

        Reterns:
            augmented_X(ndarray): 画像データのnumpy配列
            augmented_y(ndarray): ラベルデータのnumpy配列
        See:
        https://note.nkmk.me/python-opencv-warp-affine-perspective/
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
