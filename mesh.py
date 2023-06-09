import glob
import os
import numpy as np

def readlines_file(file_name):
    mesh = np.zeros(6400).reshape((100, 8, 8))
    deflist = []
    with open(file_name, 'r') as file:
        readedfile = file.readlines()
        for line in readedfile:
            words = list(line.replace("\n", ""))
            deflist.append(words)
        deflist = np.array(deflist)
        np.set_printoptions(threshold=200000000)
        print(deflist)
        for l in range(100):
            for i in range(8):#縦の8回
                for j in range(8):#横の8回,一の数でメッシュを作成
                    target = deflist[((64 * l) + (i * 8)): ((l * 64) + (i * 8 + 8)), j * 8 : j * 8 + 8] #64分割の一ブロックの抽出
                    cnt1 = np.count_nonzero(target == "1") #一ブロック内の1の数の算出
                    cnt1 = cnt1 / 64
                    mesh[l,i,j] = cnt1 #各ブロックの一の割合を代入
        return mesh
    
np.set_printoptions(threshold=2000000000)
lines = readlines_file(r"Data\hira0_00L.dat") #ファイルの読み込み
#print(lines)

if __name__ == "__main__" :
    datfiles = glob.glob("Data/*T.dat")
    meshdir =  os.path.join("mesh")
    print(datfiles)

    for datf in datfiles:
        Lmesh = readlines_file(datf) #メッシュの作成
        k = 0
        for mesh in Lmesh:
            meshpath = os.path.join(meshdir, datf.replace("Data\\",""))
            meshpath = meshpath.replace(".dat", f"_{k + 1}.npy")
            np.set_printoptions(threshold=2000)
            print(mesh)
            np.save(meshpath, mesh)
            #print(meshpath)
            k = k + 1

        