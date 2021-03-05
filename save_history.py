"""
model.fitを行ったあとに保存しておき別の場所(例えばjupyter notebook)で利用出来るようにする
保存された物はpickle.loadで利用可能になる
"""
# coding: utf-8
import pickle


def save_history(history, file_name='trainHistoryDict'):
    """
    トレーニング後のhitoryを保存する(オブジェクトを直列化)
    保存した物を使う時は
    with open(file_name, 'rb') as f:
        history = pickle.load(f)

    Args:
        hidtory(dictionary): 学習済みのhistory
        file_name(string): 保存したいファイル名
    """
    with open(file_name, 'wb') as f:
        pickle.dump(history.history, f)
