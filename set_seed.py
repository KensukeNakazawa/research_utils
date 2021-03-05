# coding: utf-8

import os
import random

import numpy as np
import tensorflow as tf


def set_random_seed(seed=19990119):
    """
    乱数を固定する

    Args:
        seed(int): 乱数シード
    """
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # tf.random.set_seed(seed)
    # session_conf = tf.compat.v1.ConfigProto(
    #     intra_op_parallelism_threads=1,
    #     inter_op_parallelism_threads=1
    # )
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.compat.v1.keras.backend.set_session(sess)

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )

    tf.set_random_seed(7)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
