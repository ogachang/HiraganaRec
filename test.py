import glob
import random
import numpy as np
from neuralnet import NeuralNet
from test_neural import TestNeural
import tqdm

teacher = np.identity(20)
weight_f = np.load("WeightData\hira_weight_f_1.npy") #trainで作った重み
weight_s = np.load("WeightData\hira_weight_s_1.npy")
test = TestNeural(weight_f, weight_s, 0.1, 0.01)
input_mesh = glob.glob("mesh/hira*T*.npy")
random.shuffle(input_mesh) #学習用データのリスト

cnt = 0
for mesh in input_mesh:
    kekka = test.forward(mesh)

    teach_number = int(mesh[11:13])
    judge = test.check(teacher[teach_number]) #正解の文字の決定
    if(judge == "correct"):
        cnt += 1
    print(kekka)
accuracy = cnt / len(input_mesh) * 100
print(accuracy)
print(test.miss)
print(test.miss_cnt)