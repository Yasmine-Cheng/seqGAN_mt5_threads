# -*- coding: utf-8 -*-
"""
配置文件 - 添加T5相關配置與記憶體優化設置
"""
import torch
import os
from datetime import datetime

# 啟用內存優化設置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

PATH = '../data/'
MAXINT = 10000
SEQ_LENGTH = 128  # 減少序列長度以節省記憶體
EMB_SIZE = 32
GENERATE_NUM = 50  # 減少生成數量
FILTER_SIZE = list(range(1,SEQ_LENGTH))
NUM_FILTER =  ([100] + [200] * 9 + [160] * SEQ_LENGTH)[0:SEQ_LENGTH-1]
DIS_NUM_EPOCH = 30  # 減少訓練輪次
DIS_NUM_EPOCH_PRETRAIN = 20
GEN_NUM_EPOCH = 30
GEN_NUM_EPOCH_PRETRAIN = 20
GEN_HIDDEN_DIM = 48
ROLLOUT_ITER = 24  # 減少rollout迭代次數
TOTAL_BATCH = 50  # 減少總批次

# T5相關配置
T5_MODEL_NAME = "uer/t5-base-chinese-cluecorpussmall"  # 使用的T5模型
T5_VOCAB_SIZE = 32128  # T5-base-chinese的詞彙表大小
T5_LEARNING_RATE = 5e-6  # 降低學習率有助於節省記憶體
T5_TASK_PREFIX = "生成討論回應: "  # 用於T5輸入的任務前綴
T5_SUMMARY_LENGTH = 50  # 減少文章摘要的最大長度 
MAX_WINDOW_SIZE = 2  # 減少滑動窗口大小

# 記憶體優化配置
USE_FP16 = True  # 啟用半精度訓練
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累積步數
DYNAMIC_BATCH_SIZE = True  # 啟用動態批次大小

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NrGPU = 0

def openLog(filename='record.txt'):
    if os.path.exists(PATH+filename):
        append_write = 'a'
    else:
        append_write = 'w'
    log = open(PATH+filename, append_write)
    return log

if DEVICE.type == 'cuda':
    NrGPU = torch.cuda.device_count()
    print('number of GPUs available:{}'.format(NrGPU))
    log = openLog('gpu.txt')
    log.write('datetime:{}, device name:{}\n'.format(datetime.now(),
                                          torch.cuda.get_device_name(0)))
    log.write('Memory Usage:')
    log.write('\nAllocated:'+str(round(torch.cuda.memory_allocated(0)/1024**3,1))+'GB')
    log.write('\nCached:   '+str(round(torch.cuda.memory_cached(0)/1024**3,1))+'GB')
    log.close()