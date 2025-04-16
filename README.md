# 🧠 Student Depression Prediction Using Machine Learning

This project aims to predict depression among students using supervised machine-learning techniques. The dataset includes demographic, academic, and lifestyle-related attributes, and the goal is to classify whether a student is likely to be depressed or not.

---

## 📂 Dataset Overview

- **Source**: Kaggle (Fetched via Kaggle API)  
- **Format**: CSV  
- **Rows**: Each row represents a unique student  
- **Target Column**: `Depression` (Yes/No)  
- **Features**:
  - Age, Gender, CGPA
  - Sleep Duration, Work Pressure, Academic Pressure
  - Study & Job Satisfaction, Diet, City, Profession, etc.

> 🔒 Ethical note: Data is anonymized. Always consider privacy when working with sensitive mental health data.

---

## 🎯 Project Objectives

- Understand relationships between lifestyle, academics, and depression.
- Clean, preprocess, and visualize the data.
- Train and compare various classification models.
- Evaluate models using accuracy, ROC-AUC, and other metrics.
- Perform feature selection and hyperparameter tuning for optimal performance.

---

## 🔧 Tech Stack

- **Languages**: Python  
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`
- **Models Used**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (K-NN)

---

## 📊 ML Workflow

1. **Data Preprocessing**  
   - Handling missing values  
   - Encoding categorical variables  
   - Feature scaling (for specific models)

2. **Exploratory Data Analysis (EDA)**  
   - Visualizing distributions  
   - Correlation heatmaps  
   - Target balance analysis

3. **Feature Selection**  
   - Using Random Forest’s feature importance  
   - Threshold-based filtering

4. **Model Training & Evaluation**  
   - Base model training  
   - Accuracy, Confusion Matrix, ROC-AUC  
   - Comparison across all models

5. **Hyperparameter Tuning**  
   - `RandomizedSearchCV` with cross-validation  
   - Optimized training and testing performance

6. **Model Evaluation**  
   - ROC Curve comparison  
   - AUC score visualization

---

## 📌 Key Results

| Model               | Tuned Testing Accuracy | AUC Score |
|--------------------|------------------------|-----------|
| Random Forest       | ✅ *Your Score*         | *Your AUC*|
| Gradient Boosting   | ✅ *Your Score*         | *Your AUC*|
| Logistic Regression | ✅ *Your Score*         | *Your AUC*|
> 🔍 Use `tuned_df` and ROC-AUC outputs to fill in the scores.

---

## 🧪 Final Conclusion

The models were able to accurately predict depression among students using behavioral and academic factors. The best-performing model based on accuracy and AUC was **Random Forest** (or your top model), indicating strong predictive power for this task. The project highlights how ML can contribute to mental health awareness through early risk identification.

---

## 📁 Folder Structure

```bash
.
├── Student Mental Health ML.ipynb       # Main notebook
├── depression_data.csv          # Dataset (if included)
├── README.md                    # Project overview
└── requirements.txt             # Required packages
```

---

## 🚀 Future Improvements

- Deploy the model using Flask/Streamlit
- Build an interactive dashboard
- Explore Deep Learning models
- Apply SHAP for interpretability

---

## 🤝 Acknowledgements

- Dataset: [Kaggle Student Depression Dataset](https://www.kaggle.com/)  
- Tools: Scikit-learn, Matplotlib, Pandas

---
