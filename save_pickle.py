"""
保存したいオブジェクトをpicke形式で保存
"""
# coding: utf-8
import pickle


def save_pickle(file_name='pickle_default', object=None):
    """保存したいオブジェクトをpicke形式で保存

    Args:
        file_name(str): 保存したいファイル名，拡張子はいらない
        object(obj): 保存したいオブジェクト
    """
    with open(file_name, 'wb') as f:
        pickle.dump(object, f)
