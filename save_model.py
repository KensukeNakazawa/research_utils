"""
class module
モデルを保存する
"""
# coding: utf-8
import os

def save_model(model, file_name='model'):
  """
  学習したモデルを保存する
  models.load_model('model.h5', compile=False)

  Args:
      model(object): 学習済みのモデルオブジェクト
      file_name(string, option): つけたい名前
  """
  path = os.getcwd()
  path = path + "/{}.hdf5".format(file_name)
  model.save(path)
