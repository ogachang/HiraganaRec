import numpy as np
import math
import random

class Nural:
    #シグモイド関数
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    #入力層から中間層初回
    def __middlelayerF(self, mesh:np.nadarry):
        middle = np.zeros(64)
        for l in range(64):
            sum = 0 #シグモイド関数のx
            weight = 2 * np.random.rand(8,8) - 1 #初期値となる重み（-1~1）
            for i in range(8):
                for j in range(8):
                    sum = sum + mesh[i][j]*weight[i][j]
            middle[l] = self.__sigmoid(sum) #合計値でシグモイド関数に渡す
        return middle #中間層の出力をまとめた配列
    
    #出力層初回
    def __finallayerF(self, middlelist:np.nadarry):
        final = np.zero(20) #あ～とまで判別するので20個
        for l in range(20):
            sum = 0 #シグモイド関数のx
            weight = 2 * np.random.rand(64) - 1 #初期値となる重み
            for i in range(64):
                sum = sum + middlelist[i]*weight[i]
            final[l] = self.__sigmoid(sum)
        return final
    

    def __middlelayer(self, mesh:np.nadarry, m_weight:np.nadarry):
        middle = np.zeros(64)
        for l in range(64):
            sum = 0 #シグモイド関数のx
            for i in range(64):
                sum = sum + mesh[i]*m_weight[i]
            middle[l] = self.__sigmoid(sum) #合計値でシグモイド関数に渡す
        return middle #中間層の出力をまとめた配列
    
    def __finallayer(self, middlelist:np.nadarry, f_weight:np.naddry):
        final = np.zero(20) #あ～とまで判別するので20個
        for l in range(20):
            sum = 0 #シグモイド関数のx
            for i in range(64):
                sum = sum + middlelist[i]*f_weight[i]
            final[l] = self.__sigmoid(sum)
        return final
    

    def __backprop(self, moji_true):

