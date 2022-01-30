from typing import Dict

import gym
import numpy as np
import random

import MAAC_agent2
from MEC_env import mec_def
from MEC_env import mec_env
import MAAC_agent
from print_logs import *
from Params import *

import tensorflow as tf
from tensorflow import keras
import tensorboard
import datetime
from matplotlib import pyplot as plt
import json
import time
import os

FL = True  # 控制是否联合学习的开关，默认True


# params['FL'] = FL


def run(conditions):
    sensor_num = conditions["sensor_num"]
    sample_method = conditions["sample_method"]
    np.random.seed(map_seed)
    random.seed(map_seed)
    tf.random.set_seed(rand_seed)

    # 选取GPU
    print("TensorFlow version: ", tf.__version__)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))  # 获得当前主机上特定运算设备的列表
    plt.rcParams['figure.figsize'] = (9, 9)  # 设置figure_size尺寸
    # logdir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    """初始化"""
    mec_world = mec_def.MEC_world(map_size, agent_num, sensor_num, obs_r, speed, collect_r, max_size, sensor_lam)
    env = mec_env.MEC_MARL_ENV(mec_world, alpha=alpha, beta=beta, aggregate_reward=aggregate_reward)
    # 建立模型
    MAAC = MAAC_agent2.MAACAgent2(env, TAU, GAMMA, LR_A, LR_C, LR_A, LR_C, BATCH_SIZE,
                                  map_size, Epsilon, sample_method, theOmega=FL_omega)

    """训练开始"""
    # 记录环境参数
    m_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    f = open('logs/hyperparam/%s.json' % m_time, 'w')
    params["conditions"] = conditions
    json.dump(params, f)
    f.close()

    # 记录控制台日志
    f_print_logs = PRINT_LOGS(m_time).open()
    print("运行程序：H_MAAC_run")
    print("运行程序：H_MAAC_run", file=f_print_logs)

    # 记录开始时间
    startTime = time.time()
    print("开始时间:", time.localtime(startTime))
    print("开始时间:", time.localtime(startTime), file=f_print_logs)

    # 训练过程
    MAAC.train(FL_omega, MAX_EPOCH, MAX_EP_STEPS, up_freq=up_freq, render=True, render_freq=render_freq, FL=FL)

    # 统计执行时间
    endTime = time.time()
    t = endTime - startTime
    print("开始时间:", time.localtime(startTime))
    print("结束时间:", time.localtime(endTime))
    print("运行时间(分钟)：", t / 60)
    print("开始时间:", time.localtime(startTime), file=f_print_logs)
    print("结束时间:", time.localtime(endTime), file=f_print_logs)
    print("运行时间(分钟)：", t / 60, file=f_print_logs)

    # 关闭记录控制台日志
    f_print_logs.close()


def experiment_5():
    """
    变量：数据源个数
    """
    sensor_nums = [30]
    # sample_methods = [1, 2]  # 默认方式二 # 采样方式一 1；    采样方式二 2
    # 现在用的采样方式是 1 
    sample_methods = [1]
    for sample in sample_methods:
        for i in range(len(sensor_nums)):
            conditions = {'sensor_num': sensor_nums[i], 'sample_method': sample}
            print("sensor_num:", sensor_nums[i])
            run(conditions)


def H_MAAC_run2():
    print("运行程序：H_MAAC_run2")

    """实验运行"""
    experiment_5()

H_MAAC_run2()
