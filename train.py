import glob
import os
import random
from matplotlib import pyplot as plt
import numpy as np
from neuralnet import NeuralNet
from train_neural import TrainNeural
import tqdm

random_weight_f = 2 * np.random.rand(64,64) - 1 #中間層の重み初期値
random_weight_s = 2 * np.random.rand(20,64) - 1 #出力層の重み初期値

print(random_weight_f)
print(random_weight_s)

teacher = np.identity(20) #教師データの作成、20×20の単位行列
print(teacher)

train = TrainNeural(random_weight_f, random_weight_s, 0.1, 0.01)
path = "mesh\\hira_19L_99.npy"
train.save_weight(path, "weight_f_0", "weight_s_0")

input_mesh = glob.glob("mesh/hira0*T*.npy")
print(input_mesh)
random.shuffle(input_mesh) #学習用データのリスト
print(input_mesh)
jedge = False #ロスから求める終了判定


while not jedge:
    error = 0
    cnt_correct = 0
    for mesh in tqdm.tqdm(input_mesh):
        #print(mesh)
        result = train.forward(mesh) #順伝搬
        teach_number = int(mesh[11:13]) #正解の文字の決定
        if(train.check(teacher[teach_number])): #正解の回数のカウント
            cnt_correct = cnt_correct + 1

        error = error + train.calc_error(teacher[teach_number]) #誤差の総和を計算する
        new_weight_s, new_weight_f = train.backward(teacher[teach_number]) #逆伝搬
    jedge = train.jedge(error, len(input_mesh))
    accuracy = (cnt_correct / len(input_mesh)) * 100
    train.accuracy_list.append(accuracy)
    print("--" + str(train.loss) + "--")
    print("--" + str(accuracy) + "--")

path = "mesh\\hira_19L_99.npy"
train.save_weight(path, "weight_f_1", "weight_s_1")
print(train.loss_list)
print(train.accuracy_list)

plt.figure(figsize=(8, 6)) # 図の設定
plt.plot(train.loss_list, label='loss') # 折れ線グラフ
plt.xlabel('x') # x軸ラベル
plt.ylabel('y') # y軸ラベル
plt.title('Train data', fontsize=20) # タイトル
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

plt.figure(figsize=(8, 6)) # 図の設定
plt.plot(train.accuracy_list, label='accuracy') # 折れ線グラフ
plt.xlabel('x') # x軸ラベル
plt.ylabel('y') # y軸ラベル
plt.title('Train data', fontsize=20) # タイトル
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()