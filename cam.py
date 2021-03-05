"""
CAM(Class Activation Map) がそれぞれ定義されるクラス
"""
# coding: utf-8

import pandas as pd
import numpy as np
import cv2

from tensorflow.keras import backend as K


class Cam:

    def __init__(self, model, x, layer_name):
        """
        Args:
        model(onject): 可視化したいモデル
        x(array): 可視化に利用したい画像(正規化されていない画像)
        layer_name(str): 可視化したい層の名
        Returns:
        g_cam: 影響の大きい箇所を色付けした画像(array)
        """
        self.model = model
        self.x = x
        self.layer_name = layer_name

    def grad_cam(self):
        """
        """
        # preproce
        # 本来は複数画像に対してmodel.predictを行うため、次元を拡張する必要がある
        # see https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
        X = np.expand_dims(self.x, axis=0)
        X = X.astype('float32')
        X = X / 255.0

        predictions = self.model.predict(X)
        class_idx = np.argmax(predictions[0])
        class_output = self.model.output[:, class_idx]

        # 勾配を取得
        conv_output = self.model.get_layer(self.layer_name).output
        # see https://keras.io/ja/backend/
        grads = K.gradients(class_output, conv_output)
        # model.inputを入力すると、 conv_outputとgradsを返す関数
        gradient_function = K.function([self.model.input], [conv_output, grads])

        output, grads_val = gradient_function(X)
        output, grads_val = output[0], grads[0]

        # 重みを平均化して、レイヤーのおアウトプットに乗じる
        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.dot(output, weights)

        # 画像化してヒートマップにして合成する
        cam = cv2.resize(cam, (200, 200), cv2.INNER_LINEAR)
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        # モノクロ画像に疑似的に色を付ける
        g_cam = cv2.applyColorMap(np.unit8(255 * cam), cv2.COLORMAP_JET)
        # 色をRGBに変換
        g_cam = cv2.cvtColor(g_cam, cv2.COLOR_BGR2RGB)
        # 元画像に合成
        g_cam = (np.float32(g_cam) + x / 2)

        return g_cam
