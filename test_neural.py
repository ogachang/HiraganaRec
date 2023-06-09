import glob
import os
import numpy as np
from neuralnet import NeuralNet

class TestNeural(NeuralNet):
    def __init__(self, weight_f, weight_s, learn, stable):
        super().__init__(weight_f, weight_s, learn, stable)
        self.loss = 0
        self.miss_cnt = np.zeros(20)#どの文字をミスしたか
        self.miss = [] #何をどのように間違えたか
    def check(self, teach):
        jp = ["あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち","つ","て","と"]
        output_answer = np.argmax(self.result)
        teacher_answer = np.argmax(teach)
        if(output_answer == teacher_answer):
            judge = "correct"
        else:
            judge = "incorrect"
            self.miss_cnt[output_answer] = self.miss_cnt[output_answer] + 1 #どの部分を間違えたか
            self.miss.append(jp[output_answer] + "→" + jp[teacher_answer])
        print("解析結果:" + jp[output_answer] + "正解:" + jp[teacher_answer] + "----" + judge + "----")
        return judge
    
    def calc_error(self,teach): #誤差を計算する関数
        error = (np.sum((teach - self.result) ** 2)) / 20 #教師データと結果の差を二乗した物の平均
        return error
