import torch
import os
import easydict
from GT_table import *

opt = easydict.EasyDict({
    "batchSize": 16,  # batch size
    "lr": 1e-3,  # learning rate
    "patch_size": 32,

    "start_iter": 1,  # 다시 돌릴때는 이거도 바꿔주기
    "nEpochs": 1500,  # training 횟수
    "snapshots": 50,  # weight 저장 주기

    "data_dir": "D:/Users/wooiljung/IML Projects/AC + phase/dataset/train",                    # train dataset 저장 위치
    "input_dir": "D:/Users/wooiljung/IML Projects/AC + phase/dataset/test",                    # test dataset 저장 위치

    "data_int_dir": "D:/Users/wooiljung/IML Projects/AC + phase/dataset/Amp_train",            # amplitude map 저장 위치 (train)
    "input_int_dir": "D:/Users/wooiljung/IML Projects/AC + phase/dataset/Amp_test",            # amplitude map 저장 위치 (eval)

    "output": "D:/Users/wooiljung/IML Projects/AC amp + phase X-attention/results/output maps",                    # 결과영상 저장 위치
    "save_folder": "D:/Users/wooiljung/IML Projects/AC amp + phase X-attention/results/weights/",                  # weight 저장 위치 (requires / at end)
    "trainlogfilepath": 'D:/Users/wooiljung/IML Projects/AC amp + phase X-attention/results/training loss.txt',    # train loss를 savepoint마다 저장할 위치

    "model_type": "result",  # 모델이름 (result)

    "resume": False,
    "pretrained": False,
    "data_augmentation": False,
    "residual": False,
    "gpu_mode": True,
    "threads": 1,
    "seed": 123,
    "gpus": 1,
    "test_dataset": "Test.pt",
    "testBatchSize": 1,

    "testresult": "same_frame",
    "evalmode": False,
    "evalweightpath": "D:/Users/wooiljung/IML Projects/AC amp + phase X-attention/results/weights/DESKTOP-JB0RI7F_epoch_650.pth",  # 사용할 weight
    "evallogfilepath": "D:/Users/wooiljung/IML Projects/AC amp + phase X-attention/results/eval loss cross attention.txt",         # eval 결과 저장할 .txt 파일

    "outliermode": True,
    "outlierfilepath": "D:/Users/wooiljung/IML Projects/AC amp + phase X-attention/results/outlier loss cross attention.txt",
    "outlier_list": [14, 20, 30, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 72],
})


checkpoint_name = 'D:/Max/weights/same_frame/'
checkpoint_name2 = os.path.join(checkpoint_name + 'DESKTOP-NJ37VPA_epoch_359.pth')

input1 = torch.load('D:/Users/wooiljung/IML Projects/AC + phase/dataset/ptfiles/dataset_video_scene(150)_length(10)_scale(3)_Amb_Train.pt')    # train pt 파일
test1 = torch.load('D:/Users/wooiljung/IML Projects/AC + phase/dataset/ptfiles/dataset_video_scene(75)_length(10)_scale(3)_Amb_Test.pt')       # eval pt 파일

input11 = torch.zeros(150, 27, input1.size(2), input1.size(3))
for i in range(9):
    input11[:, 3*i:3*i+3, :, :] = input1[:, 3*i:3*i+3, :, :]

test11 = torch.zeros(75, 27, test1.size(2), test1.size(3))
for i in range(9):
    test11[:, 3*i:3*i+3, :, :] = test1[:, 3*i:3*i+3, :, :]