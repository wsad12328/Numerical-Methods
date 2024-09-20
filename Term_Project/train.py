import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Add
from tensorflow.keras.utils import to_categorical
import sys
import matplotlib.pyplot as plt

def load_and_train_model(seq_length, count):
    # 載入數據
    with open(f'dataset/data_{method}.npy', 'rb') as f:
        data = np.load(f)

    # 前 seq_length 個數據作為輸入
    x_train = np.array(data[:,:seq_length]).reshape((count, seq_length, 1))
    x_train = x_train / 10
    # 後 seq_length 個數據作為目標
    y_train = np.array(data[:,seq_length:2*seq_length]).reshape((count, seq_length, 1))
    y_train = to_categorical(y_train, num_classes=10)

    # model = Sequential()
    # model.add(LSTM(512, input_shape=(seq_length, 1), return_sequences=True))
    # model.add(LSTM(128, input_shape=(seq_length, 1), return_sequences=True))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(10, activation='softmax'))
    inputs = Input(shape=(seq_length, 1))
    x = LSTM(256, return_sequences=True)(inputs)
    residual = Dense(256)(inputs)  
    x = Add()([x, residual])  # 殘差

    # 第二個 LSTM 層
    x = LSTM(256, return_sequences=True)(x)
    residual = Dense(256)(residual)  
    x = Add()([x, residual])  # 殘差

    # Dense 層
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.summary()

    # 編譯模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 訓練模型
    history = model.fit(x_train, y_train, epochs=120, batch_size=200)

    # 保存模型
    model.save(f'model/lstm_model_{method}.h5')

     # Plot training loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.plot(np.array(history.history['accuracy']))

    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.plot(history.history['loss'])

    plt.tight_layout()
    plt.savefig(f'training_plot_{method}.png')  # Save plot as PDF

if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] not in ['-m', '--method']:
        print("Usage: python script.py -m <random/lcg> <number_of_sequences> <length_of_each_sequence>")
        sys.exit(1)
    else:
        method = sys.argv[2]
        count = int(sys.argv[3])
        seq_length = int(sys.argv[4])
            
        # generate_data(count, seq_length * 2)  # 生成數據，每筆數據長度為 2 * seq_length
        load_and_train_model(seq_length, count)
