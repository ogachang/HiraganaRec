import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from neuralnet import NeuralNet

class TrainNeural(NeuralNet):
    def __init__(self, weight_f, weight_s, learn, stable):
        super().__init__(weight_f, weight_s, learn, stable)
        self.loss = 0
        self.loss_list = []
        self.accuracy_list = []


    def calc_error(self,teach): #誤差を計算する関数
        error = (np.sum((teach - self.result) ** 2)) / 20 #教師データと結果の差を二乗した物の平均
        return error
    
    def check(self,teach):
        output_answer = np.argmax(self.result)
        teacher_answer = np.argmax(teach)
        if(output_answer == teacher_answer):
            return True
        else:
            return False        
    
    #入力パターンn=2000、各パターンｐにおいての20個の出力を教師データと比較する　引数は二乗誤差平均の総和
    def jedge(self, total_error, cnt_input):
        jedge = False #継続かどうかの判断
        gosa = total_error / cnt_input
        self.loss = gosa
        self.loss_list.append(gosa) #誤差をリストに格納
        if(gosa < 0.01):
            jedge = True
        else:
            jedge = False
        return jedge
    
    def save_weight(self, meshpath, f_savename, s_savename):
        weightdir =  os.path.join("WeightData")
        weight_pf = os.path.join(weightdir, (meshpath[5:10] + f_savename + ".npy"))
        weight_ps = os.path.join(weightdir, (meshpath[5:10] + s_savename + ".npy"))
        np.save(weight_pf, self.weight_f)
        np.save(weight_ps, self.weight_s)
        return