import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('C:/Users/mayan/OneDrive/Desktop/diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model trained and saved successfully!")
print(f"Training accuracy: {model.score(X_train_scaled, y_train):.3f}")
print(f"Testing accuracy: {model.score(X_test_scaled, y_test):.3f}")

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    return prediction[0], probability[0]

print("\n--- Example Prediction ---")
result, prob = predict_diabetes(6, 148, 72, 35, 0, 33.6, 0.627, 50)
print(f"Input: Pregnancies=6, Glucose=148, BP=72, SkinThickness=35, Insulin=0, BMI=33.6, DPF=0.627, Age=50")
print(f"Prediction: {'Diabetic' if result == 1 else 'Not Diabetic'}")
print(f"Probability: No Diabetes={prob[0]:.2%}, Diabetes={prob[1]:.2%}")
