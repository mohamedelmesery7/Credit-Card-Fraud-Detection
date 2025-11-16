# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using various machine learning techniques. The primary challenge is the highly imbalanced nature of the dataset, where fraudulent transactions are a very small minority.

## 📖 Table of Contents
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Imbalance Handling Techniques](#-imbalance-handling-techniques)
- [Models Evaluated](#-models-evaluated)
- [Results](#-results)
- [How to Run](#-how-to-run)

## 📊 Dataset

The dataset used is `creditcard.csv`, which contains credit card transactions made over a period of two days. It consists of 31 features, including `Time`, `Amount`, and 28 anonymized features (`V1` to `V28`) obtained through PCA. The target variable is `Class`, which is `1` for fraudulent transactions and `0` for legitimate ones.

- **Class Imbalance:** The dataset is severely imbalanced, with fraudulent transactions accounting for only about 0.17% of all transactions.

## ⚙️ Project Workflow

1.  **Data Loading and Cleaning:** The dataset is loaded, and duplicate entries are removed.
2.  **Exploratory Data Analysis (EDA):** The class distribution is visualized to understand the extent of the imbalance.
3.  **Data Splitting:** The data is split into training and testing sets using a stratified split to maintain the class proportion in both sets.
4.  **Preprocessing:**
    - The `Amount` feature is log-transformed (`np.log1p`) to handle its skewed distribution.
    - The `Time` and `Amount` features are scaled using `StandardScaler`.
5.  **Model Training and Evaluation:** Various techniques for handling class imbalance are applied, and multiple models are trained and evaluated.

## ⚖️ Imbalance Handling Techniques

Several methods were implemented and compared to address the class imbalance:

1.  **Random Undersampling:** A balanced subset of the data was created by randomly selecting non-fraudulent transactions to match the number of fraudulent ones. This method led to significant information loss and poor model performance.
2.  **Cost-Sensitive Learning (`class_weight='balanced'`):** This approach adjusts the model's learning process to penalize misclassifications of the minority class more heavily. It was applied to both Logistic Regression and Random Forest models.
3.  **Oversampling (SMOTE):** The Synthetic Minority Over-sampling Technique (SMOTE) was used to generate synthetic samples for the minority class. This was implemented within an `imblearn.pipeline` to prevent data leakage during cross-validation.
4.  **Undersampling (NearMiss):** An alternative undersampling technique that selects samples from the majority class based on their distance to minority class samples. This also resulted in suboptimal performance compared to other methods.


## 🤖 Models Evaluated

- **Logistic Regression**
- **Random Forest Classifier**

## 📈 Results

The evaluation was performed using 5-fold stratified cross-validation, with `classification_report`, `roc_auc_score`, and `confusion_matrix` as the primary metrics.

The two most effective approaches were:

1.  **Random Forest with `class_weight='balanced'`:** This model achieved excellent recall for the fraud class without significantly compromising precision.
2.  **Random Forest with SMOTE:** This combination also performed very well, showing a strong ability to identify fraudulent transactions.

A final comparison showed that both models are viable solutions, with a slight trade-off between precision and recall. The choice between them would depend on the specific business objective (e.g., minimizing false negatives vs. minimizing false positives).

## 🚀 How to Run

1.  **Prerequisites:** Ensure you have Python and Jupyter Notebook installed.
2.  **Install Libraries:** Open the `Credit fraud.ipynb` notebook and run the first few cells to install the required libraries like `pandas`, `scikit-learn`, `imblearn`, `matplotlib`, `seaborn`.
3.  **Execute Notebook:** Run the cells in the notebook sequentially to reproduce the analysis and results.
