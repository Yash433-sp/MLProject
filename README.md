# Loan Approval Prediction System Project
This project is a Machine Learning-based Loan Approval Prediction System that helps financial institutions determine whether a loan application should be approved or rejected based on applicant details. The system uses various machine learning models to predict loan approval status with high accuracy.
# Objective
The main objective of this project is to develop a machine learning model that can accurately predict loan approval based on applicant details. By leveraging data-driven insights, the system aims to:
Reduce manual loan processing time.
Minimize loan default risk.
Improve decision-making for financial institutions.
# Dataset Used
The dataset consists of multiple features related to loan applications, including:
No. of Dependents
Education Level
Self-Employment Status
Annual Income
Loan Amount Requested
Loan Term
CIBIL Score
Asset Values (Residential, Commercial, Luxury, Bank Assets)
Loan Status (Target Variable)
# Model Implemented
Several models were trained and compared to determine the best-performing one:
Linear Regression
Random Forest Classifier ✅ (Best Performing Model)
K-Nearest Neighbors (KNN)
Decision Tree Classifier
Support Vector Machine (SVM)
The best-performing model was saved using joblib for real-time predictions.
# Performance Metrics
To evaluate model performance, the following metrics were used:
Accuracy: Measures the overall correctness of the model.
Precision: Measures how many of the predicted positive instances were actually positive.
Recall: Measures how many of the actual positive instances were correctly identified.
F1 Score: Harmonic mean of precision and recall.
Confusion Matrix: Provides a detailed breakdown of true/false positives and negatives.
# Installation And Run
To set up and run the project on your local machine:
 Clone the repository:git clone https://github.com/your-username/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
 Install dependencies:pip install -r requirements.txt
 # Challenges
Handling categorical variables using Label Encoding.
Feature scaling for numerical values.
Addressing imbalanced data for better predictions.
Optimizing hyperparameters to improve accuracy.
# Learnings
Feature engineering plays a key role in improving model performance.
Random Forest is effective for handling complex datasets.
Streamlit is a great tool for deploying ML models interactively.
