# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 중요 지침 (Important Guidelines)

**언어 사용 규칙**: 이 프로젝트에서 작업할 때는 **모든 답변과 설명을 한글로** 제공해야 합니다. 이는 한국어 사용자와의 원활한 소통을 위함입니다.

## Project Overview

This is a machine learning-based earthquake detection system that distinguishes real earthquakes from false positives (industrial vibrations and daily life activities) using ConvLSTM deep learning models. The system achieves 98.5% accuracy on a 3-class classification problem with balanced datasets.

## Common Development Commands

### Model Training
```bash
# Train the main NEW2 ConvLSTM 3-class model
python train_new2_convlstm.py

# Train LSTM model for comparison
python train_lstm_3class_model.py

# Train 4-class variant
python train_convlstm_4class.py
```

### Data Processing
```bash
# Create balanced 3-class dataset from NEW2 data
python 3class_balanced_dataset.py

# Process individual data sources
python new2_process_kor_quake_data.py  # Korean earthquake data
python new2_process_jpn_quake_data.py  # Japanese earthquake data
python new2_process_industry_data.py   # Industrial vibration data
python new2_process_motor_data.py      # Motor vibration data
python new2_process_live_data.py       # Living activity data
python new2_process_irregular_data.py  # Irregular vibration data
```

### Real-time Inference
```bash
# Run real-time earthquake detection
python realtime_inference_convlstm.py

# Alternative real-time detection system
python new_realtime_inference_convlstm.py
```

### Data Analysis
```bash
# Analyze class patterns and similarities
python analyze_class_patterns.py

# Analyze false positive patterns
python analyze_false_positive_patterns.py

# Test trained models on new datasets
python test_new_motor_dataset.py
python test_kma_earthquake_dataset.py
```

### Web Interface
```bash
# Launch Streamlit monitoring dashboard
streamlit run streamlit_seismic_app.py
```

## Architecture Overview

### Data Pipeline Architecture
1. **Raw Data Sources**: 
   - Korean KMA earthquake data (2,308 samples)
   - Japanese Ohtashi earthquake data (1,564 samples) 
   - Industrial machinery vibrations (1,110 samples)
   - Motor operations (1,604 samples)
   - Living activities (2,135 samples)
   - Irregular urban activities (700 samples)

2. **Data Format**: 40-second time windows (4000 samples @ 100Hz) with 3-axis accelerometer data
   - Input shape: `(N, 40, 3, 100, 1)` for ConvLSTM
   - Classes: 0=지진(earthquake), 1=규칙적산업진동(regular industrial), 2=불규칙생활진동(irregular living)

3. **Preprocessing**: Z-score normalization, 60/20/20 train/val/test split

### Model Architecture
- **Primary Model**: ConvLSTM (Convolutional LSTM) for spatiotemporal pattern recognition
- **Comparison Models**: Standard LSTM, Random Forest, SVM, Logistic Regression
- **Best Performance**: 98.5% test accuracy with balanced precision/recall across all classes

### Real-time System Architecture
- **Data Source**: InfluxDB time-series database
- **Processing**: Real-time data preprocessing and model inference
- **Output**: Classification results with confidence scores
- **Monitoring**: Streamlit web dashboard for live monitoring

## Key File Mappings

### Core Models
- `train_new2_convlstm.py`: Main ConvLSTM training script (current best)
- `enhanced_convlstm_model.py`: Enhanced model with traditional ML comparison
- `convlstm_model.py`: Original ConvLSTM implementation

### Data Processing
- `3class_balanced_dataset.py`: Creates balanced 3-class dataset from NEW2 data
- `new2_process_*.py`: Individual data source processors for NEW2 dataset
- `combine_4class_dataset.py`: Combines data for 4-class experiments

### Real-time Systems  
- `realtime_inference_convlstm.py`: Production real-time detection system
- `streamlit_seismic_app.py`: Web monitoring dashboard
- `influx_base.py`: InfluxDB integration utilities

### Analysis Tools
- `analyze_class_patterns.py`: Pattern analysis and visualization
- `analyze_false_positive_patterns.py`: False positive analysis
- `test_*.py`: Model evaluation scripts

## Data Structure

### Training Data Location
- `new2_3class_dataset/`: Balanced 3-class dataset (current production)
- `*.npy` files: NumPy arrays for different data sources and classes
- `models/`: Saved trained models
- `data/`: Raw data sources (KMA, industry, irregular)

### Key Configuration Files
- `available_data_map.json`: Data inventory and statistics
- `new2_training_summary.json`: Latest model performance results
- `preprocessing_metadata_*.json`: Data preprocessing parameters

## Development Notes

### Data Preprocessing Standards
- All sensor data must be 3-axis (X, Y, Z) accelerometer readings
- 100Hz sampling rate required for compatibility
- 40-second windows (4000 samples) for ConvLSTM input
- Z-score normalization applied consistently

### Model Performance Targets
- Target accuracy: >98% on balanced test set
- Class balance requirement: Equal representation across all 3 classes
- Real-time inference: <1 second processing time per window

### Real-time System Requirements
- InfluxDB connection for live data streaming
- Model loading time optimization for production deployment
- Confidence threshold tuning for alert systems

### Testing Protocols
- Use balanced test sets for fair evaluation
- Cross-validation with time-series splits to prevent data leakage
- Performance monitoring on new/unseen data sources