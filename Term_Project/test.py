import numpy as np
from tensorflow.keras.models import load_model
import sys

# 設置 LCG 參數
a = 1664525
c = 1013904223
m = 2**32

# LCG 生成數據
def lcg(length, seed, a, c, m):
    data = np.zeros(length, dtype=int)
    x = seed
    for j in range(length):
        x = (a * x + c) % m
        data[j] = x % 10
    return data

def generate_data(n, length):
    # 生成 n 個不同的 seed
    seeds = np.random.randint(0, m, size=n)

    # 生成 n 筆長度為 length 的數據，每個使用不同的 seed
    data = np.array([lcg(length, seed, a, c, m) for seed in seeds])

    return data

def test_model(seq_length, test_count):
    # 生成測試數據
    test_data = generate_data(test_count, seq_length * 2)

    # 前 seq_length 個數據作為輸入
    x_test = np.array(test_data[:,:seq_length]).reshape((test_count, seq_length, 1))
    x_test = x_test / 10
    # 後 seq_length 個數據作為目標
    y_true = np.array(test_data[:,seq_length:2*seq_length]).reshape((test_count, seq_length))

    # 載入模型
    model = load_model(f'model/lstm_model_{method}.h5')

    # 預測
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=-1)

    # 計算準確度
    accuracy = np.mean(y_pred == y_true)
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] not in ['-m', '--method']:
        print("Usage: python script.py -m <random/lcg> <number_of_sequences> <length_of_each_sequence>")
        sys.exit(1)
    else:
        method = sys.argv[2]
        test_count = int(sys.argv[3])
        seq_length = int(sys.argv[4])
        
        test_model(seq_length, test_count)
