# 💳 Credit Card Fraud Detection

This project is about detecting fraud in credit card transactions using a simple machine learning model called **Logistic Regression**.  
The goal is to tell whether a transaction is **genuine (0)** or **fraudulent (1)** based on the data.

---

## 📁 Dataset Information

We are using a public dataset that has credit card transactions from European cardholders in September 2013.  
It contains **284,807 transactions**, and only **492 of them are fraud** — so the data is very imbalanced.

Each transaction has:

- `Time`: seconds passed since the first transaction  
- `V1` to `V28`: features transformed using PCA (for privacy)  
- `Amount`: transaction amount  
- `Class`: the label → `0` for legit, `1` for fraud

🔗 **Download the dataset from here**:  
[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> ❗ After downloading, place the `creditcard.csv` file in your project folder to run the notebook.

---

## 📝 Steps Followed in the Project

### 1. **Imported Required Libraries**
We used Python libraries like:
- `pandas` and `numpy` to handle data
- `scikit-learn` to train and test our model

### 2. **Read the Dataset**
Loaded the data from `creditcard.csv` and checked its structure:
- Viewed the first few rows
- Checked for missing values
- Counted how many are legit vs fraud

### 3. **Separated Fraud and Legit Transactions**
The dataset had:
- Around 2.8 lakh legit transactions  
- Only 492 fraud transactions

We used **undersampling** to balance the data by taking only 492 legit transactions to match the 492 fraud ones.

### 4. **Combined the Balanced Data**
Created a new dataset with:
- 492 fraud transactions  
- 492 sampled legit transactions

So now we have a balanced dataset with 984 rows.

### 5. **Split Data for Training and Testing**
Split the data into:
- **Features (X)** → the input values
- **Labels (Y)** → fraud or not

Then divided it into:
- 80% training data  
- 20% testing data

### 6. **Trained the Model**
Used **Logistic Regression** to train the model using the training data.

### 7. **Checked the Accuracy**
Checked how well the model performs on both:
- Training data
- Testing data

---
