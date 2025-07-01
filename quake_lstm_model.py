from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(64, return_sequences=False, input_shape=(40, 300)))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 이진 분류 출력

# 컴파일 설정
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 구조 요약 출력
model.summary()