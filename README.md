# Diabetes Prediction System

A machine learning project that predicts diabetes using patient health metrics.

## Dataset

The project uses `diabetes.csv` containing 768 patient records with 8 features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

## Files

- `diabetes.csv` - Dataset
- `diabetes_prediction.py` - Model training script
- `predict_new.py` - Interactive prediction script
- `requirements.txt` - Dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the Model
```bash
python diabetes_prediction.py
```

### Make Predictions
```bash
python predict_new.py
```

Enter patient details when prompted to get diabetes prediction and probability.

## Model

- Algorithm: Random Forest Classifier
- Features: 8 health metrics
- Preprocessing: Standard Scaling
- Train/Test Split: 80/20

## Output

The model provides:
- Binary prediction (Diabetic/Not Diabetic)
- Probability scores for both classes
