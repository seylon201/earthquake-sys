
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization,
    Add, Activation
)

class SimpleEQTransformer:
    """EQTransformer 대안 모델 (Transformer-inspired)"""
    
    def __init__(self, input_shape=(6000, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def create_model(self):
        """Transformer-inspired 모델 생성"""
        inputs = Input(shape=self.input_shape, name='input_waveform')
        
        # 1D CNN으로 지역 특징 추출
        x = Conv1D(64, 15, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(64, 15, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        # 다운샘플링으로 시퀀스 길이 줄이기
        x = Conv1D(128, 5, strides=4, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv1D(128, 5, strides=4, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        # LSTM으로 시계열 패턴 학습
        x = LSTM(256, return_sequences=True, dropout=0.3)(x)
        x = LSTM(128, return_sequences=True, dropout=0.3)(x)
        
        # Self-Attention (Transformer의 핵심)
        attention_output = MultiHeadAttention(
            num_heads=8, 
            key_dim=64,
            name='self_attention'
        )(x, x)
        
        # Residual connection
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        
        # 추가 처리층
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # 최종 분류층
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # 출력층
        outputs = Dense(self.num_classes, activation='softmax', name='classification')(x)
        
        # 모델 생성
        self.model = Model(inputs=inputs, outputs=outputs, name='SimpleEQTransformer')
        
        return self.model
    
    def compile_model(self, learning_rate=1e-4):
        """모델 컴파일"""
        if self.model is None:
            self.create_model()
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

# 사용 예시
if __name__ == "__main__":
    # 모델 생성 테스트
    eqt_alternative = SimpleEQTransformer(input_shape=(6000, 3), num_classes=3)
    model = eqt_alternative.compile_model()
    
    print("✅ EQTransformer 대안 모델 생성 완료!")
    print(f"📊 모델 구조: {model.count_params():,} 파라미터")
    model.summary()
