# Problem Statement

## Objective

Develop a machine learning system to predict the likelihood of diabetes in patients based on diagnostic measurements.

## Background

Diabetes is a chronic disease affecting millions worldwide. Early detection enables timely intervention and better health outcomes. This project aims to automate diabetes risk assessment using historical patient data.

## Problem

Given patient health metrics, predict whether the patient has diabetes or not.

## Input Features

1. **Pregnancies** - Number of times pregnant
2. **Glucose** - Plasma glucose concentration
3. **Blood Pressure** - Diastolic blood pressure (mm Hg)
4. **Skin Thickness** - Triceps skin fold thickness (mm)
5. **Insulin** - 2-Hour serum insulin (mu U/ml)
6. **BMI** - Body mass index (weight in kg/(height in m)^2)
7. **Diabetes Pedigree Function** - Genetic influence score
8. **Age** - Age in years

## Output

Binary classification:
- 0: No diabetes
- 1: Diabetes present

## Approach

1. Load and preprocess the diabetes dataset
2. Split data into training and testing sets
3. Apply feature scaling using StandardScaler
4. Train Random Forest Classifier
5. Evaluate model performance
6. Save trained model for predictions
7. Provide interactive interface for new predictions

## Success Criteria

- Accurate predictions on test data
- Probability scores for risk assessment
- User-friendly prediction interface
