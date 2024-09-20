import numpy as np
import sys
import os

# 設置 LCG 參數
a = 2**16 + 3 
c = 0
m = 2**31

# LCG 生成數據
def lcg(length, seed, a, c, m):
    data = np.zeros(length, dtype=int)
    x = seed
    for j in range(length):
        x = (a * x + c) % m
        data[j] = x % 10
    return data

# /dev/random 生成數據
def random_data(length, count):
    random_numbers = []
    with open("/dev/random", "rb") as f:
        for _ in range(count):
            random_bytes = f.read(length)
            random_integers = [int(byte) % 10 for byte in random_bytes]  # 將每個字節轉換為0到9的整數
            random_numbers.append(random_integers)
    return random_numbers

def generate_data(method, n, length):
    if method == 'random':
        return random_data(length, n)
    elif method == 'lcg':
        # 生成 n 個不同的 seed
        seeds = np.random.randint(0, m, size=n)
        # 生成 n 筆長度為 length 的數據，每個使用不同的 seed
        return np.array([lcg(length, seed, a, c, m) for seed in seeds])
    else:
        print("Invalid method. Please choose 'random' or 'lcg'.")
        sys.exit(1)

def main():
    if len(sys.argv) != 5 or sys.argv[1] not in ['-m', '--method']:
        print("Usage: python script.py -m <random/lcg> <number_of_sequences> <length_of_each_sequence>")
        sys.exit(1)

    method = sys.argv[2]
    n = int(sys.argv[3])
    length = int(sys.argv[4])
    data = generate_data(method, n, length)

    # 保存數據
    with open(f'dataset/data_{method}.npy', 'wb') as f:
        np.save(f, data)

if __name__ == "__main__":
    main()
