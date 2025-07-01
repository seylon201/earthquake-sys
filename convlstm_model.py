import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ✅ 데이터 로드
X = np.load('X_total.npy')      # shape: (5236, 40, 3, 100, 1)
y = np.load('y_total.npy')      # shape: (5236,) with values 0,1,2,3

# ✅ 이진 분류 라벨 변환 (2,3 → 1)
y = np.where(y >= 2, 1, 0)

# ✅ 정규화
X = (X - np.mean(X)) / np.std(X)

# ✅ 데이터 분할
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# ✅ ConvLSTM 이진 분류 모델
def build_convlstm_binary():
    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(1, 3), activation='tanh',
                         padding='valid', input_shape=(40, 3, 100, 1), return_sequences=False))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

model = build_convlstm_binary()

# ✅ 학습
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1)

# ✅ 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")

# ✅ 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy"); plt.grid(); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss"); plt.grid(); plt.legend()

plt.suptitle("Binary ConvLSTM Training Results (논문 기준 5236건)")
plt.tight_layout()
plt.savefig("convlstm_binary_final_training_results.png")
plt.show()