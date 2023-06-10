# HiraganaRec
ひらがなの判別プログラムです。入力に指定のメッシュを入れるとあ～とまでの文字を判別します。

# 使用方法
学習方法

train.pyを使用してください。
インスタンス：train = TrainNeural(random_weight_f, random_weight_s, 0.1, 0.01)
第1，2引数にそれぞれ中間層、出力層の重みを入れる（基本的には64×64,20×64の-1~1の範囲をとるランダムな行列を入れてください）。
第3、4引数には学習定数と安定化定数を入れる（とりあえず0.1,0.01を入れていますが好みによって変えてください）。

train.pyのinput_mesh変数のglob関数の引数を学習したいファイルが全て取れるような名前に変える。
例：筆記者0Lなら"mesh/hira0*L*"

重みの初期値と、最終の重みを保存する際は関数の引数を変える。
例：train.save_weight(path, "weight_f_1", "weight_s_1")
3カ所を設定したら学習可能です。

判定
test.pyを使用してください。
学習で保存した重みを以下の2カ所にパスを書き込んでロードする。
weight_f = np.load("WeightData\hira_weight_f_1.npy") 　中間層
weight_s = np.load("WeightData\hira_weight_s_1.npy")　出力層

インスタンス：train = TrainNeural(random_weight_f, random_weight_s, 0.1, 0.01)
第1，2引数にそれぞれ中間層、出力層の重みを入れる
第3、4引数には学習定数と安定化定数を入れる

input_mesh変数のglob関数の引数を学習したいファイルが全て取れるような名前に変える。

例：筆記者0Tなら"mesh/hira0*T*"

3カ所設定したら実行可能です。
