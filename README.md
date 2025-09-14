# Fraud Detection with PySpark

This project demonstrates how to build a **fraud detection model** using **Apache Spark (PySpark)** and **Machine Learning (MLlib)**.  
The dataset is highly **imbalanced** (fraud cases are very rare compared to normal transactions), so techniques like **class weighting** are applied.  

---

## Dataset

The dataset contains financial transactions with the following columns:

| Column          | Description |
|-----------------|-------------|
| **step**        | Time step in hours (unit of time). |
| **type**        | Type of transaction (e.g., PAYMENT, TRANSFER, CASH_OUT). |
| **amount**      | Transaction amount. |
| **nameOrig**    | Customer initiating the transaction (anonymized). |
| **oldbalanceOrg** | Initial balance of origin account. |
| **newbalanceOrig** | New balance of origin account after transaction. |
| **nameDest**    | Recipient of the transaction (anonymized). |
| **oldbalanceDest** | Initial balance of destination account. |
| **newbalanceDest** | New balance of destination account after transaction. |
| **isFraud**     | Target label (1 = Fraud, 0 = Legit). |
| **isFlaggedFraud** | Flagged transactions by system (rule-based). |

## Steps

1. **Data Loading**  
   Load dataset into Spark DataFrame.

2. **Exploratory Data Analysis (EDA)**  
   - Checked for missing values.  
   - Observed strong **class imbalance** (fraud << non-fraud).

3. **Class Weighting**  
   - Added a `classWeightCol` column to balance fraud vs non-fraud samples during training.

4. **Feature Engineering**  
   - Dropped string columns (`nameOrig`, `nameDest`) not useful for ML.  
   - Used `VectorAssembler` to combine numerical features into a `features` column.

5. **Train/Test Split**  
   - 80% for training, 20% for testing.

6. **Model Training**  
   - Logistic Regression (`pyspark.ml.classification.LogisticRegression`)  
   - Used weighted loss function to handle imbalance.

7. **Evaluation**  
   - Evaluated with **AUC (Area Under ROC Curve)**.  
   - AUC is preferred over accuracy in imbalanced datasets.

## Tech Stack

- **Apache Spark (PySpark MLlib)**  
- **Python 3**  
- **Jupyter Notebook**  
- **Dataset: Fraud Transaction Dataset (Kaggle)**  

## Results

- Model was evaluated using **AUC (Area Under ROC)**.  
- AUC > 0.8 indicates a **good fraud detection model**.  
- Helps in identifying fraudulent transactions with higher reliability than just accuracy.
