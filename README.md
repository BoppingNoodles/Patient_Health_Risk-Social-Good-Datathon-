Public Health Risk: From Vitals to Intervention
Developed for the 7th Annual Datathon for Social Good By Caden Luu

📌 Project Overview
This project was developed during the 7th Annual Datathon for Social Good to address critical challenges in healthcare resource allocation. Using a dataset of 1,000 anonymized, real-world patient records, the project aims to create a predictive tool that identifies patient health risk levels to enable early clinical intervention.

The primary goal is to provide a data-driven approach to medical prioritization without compromising sensitive patient information.

🚀 Key Features
Risk Classification: Categorizes patients into Low, Medium, or High-risk levels based on vital signs.

Clinical Feature Engineering: Identifies "Fever" states by applying clinical thresholds (e.g., Temperature > 100.4°F and Heart Rate > 95 bpm) to enhance prediction accuracy.

Data Reliability: Includes a rigorous preprocessing pipeline that handles duplicate checks and missing value verification to ensure data integrity.

📊 Dataset Detail
Source: 1,000 anonymized real-world patient records licensed under Apache 2.0.

Input Variables: Respiratory Rate, Oxygen Saturation, O2 Scale, Systolic Blood Pressure, Heart Rate, and Body Temperature.

Target: Risk_Level (Low, Medium, High).

🛠️ Technical Stack
Language: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn.

Machine Learning: XGBoost (utilized for advanced classification) and Scikit-Learn (for preprocessing and evaluation).

Persistence: Joblib for model serialization and metadata storage.

💻 Workflow Summary
Preprocessing: Dropped non-predictive identifiers (Patient_ID) and standardized units, such as converting temperatures for clinical consistency.

EDA: Performed exploratory analysis to understand the distributions of vitals and their correlation with risk levels.

Modeling: Employed XGBoost for robust classification of high-dimensional health data.

Serialization: Saved the final model along with feature metadata (thresholds and unit info) to ensure the pipeline is reproducible for clinical deployment.

🌟 Datathon Goals
As part of the Datathon for Social Good, this project emphasizes:

Scalability: Building models that can be adapted to various clinical settings.

Actionability: Moving beyond theory to create tools that directly support healthcare providers.

Privacy: Demonstrating effective data science practices on anonymized datasets.
