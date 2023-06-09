import glob
import os
import numpy as np

class NeuralNet:
    def __init__(self, weight_f, weight_s, learn, stable):
        self.weight_f = weight_f
        self.weight_s = weight_s
        self.deff_f = np.zeros(4096).reshape((64,64)) #誤差逆伝搬の変化量入力→中間
        self.deff_s = np.zeros(1280).reshape((20,64)) #誤差逆伝搬の変化量中間→出力
        self.learn = learn #0.025~0.1ぐらい
        self.stable = stable #0.0005~0.01ぐらい

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, filepath): #順伝搬処理 入力→中間 引数はファイルパス、重み、重みを格納するディクショナリ
        mesh = np.load(filepath)
        mesh = np.reshape(mesh, (64))#入力値1~64に変換
        self.input = mesh
        self.intermediate = self.sigmoid(np.dot(self.weight_f, mesh)) #合計値でシグモイド関数に渡す
        self.result = self.sigmoid(np.dot(self.weight_s, self.intermediate)) #合計値でシグモイド関数に渡す
        return self.result
    
    def backward(self, teacher): #出力層の逆伝搬
        # 中間層逆伝播を計算:式(5.12)
        buf = 0
        for j in range(64):
            for i in range(64):
                for k in range(20):
                    buf = buf + (teacher[k] - self.result[k])*self.result[k]*(1 - self.result[k])*self.weight_s[k][j]
                grad_to_wf = self.learn*(1 - self.intermediate[j])*self.input[i]*buf + self.stable*self.deff_f[j][i]
                buf = 0 #bufの初期化
                self.deff_f[j][i] = grad_to_wf
                self.weight_f[j][i] = self.weight_f[j][i] + grad_to_wf

        # 出力層逆伝播を計算:式(5.12)
        for k in range(20):
            for j in range(64):
                grad_to_ws = self.learn*(teacher[k] - self.result[k])*self.result[k]*(1 - self.result[k])*self.intermediate[j] + (self.stable*self.deff_s[k][j])
                self.deff_s[k][j] = grad_to_ws #変化量の保存
                self.weight_s[k][j] = self.weight_s[k][j] + grad_to_ws #次回の重みに変更
        return self.weight_f, self.weight_s

    