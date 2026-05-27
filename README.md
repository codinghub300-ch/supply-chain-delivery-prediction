# Supply Chain Delivery Prediction 

A Machine Learning project developed to predict late delivery risks in supply chain operations using data preprocessing, feature engineering, and predictive modeling techniques.

---

##  Project Overview

This project analyzes supply chain shipment data to predict whether deliveries are likely to be delayed.

The system applies:
- Data Cleaning
- Feature Engineering
- Machine Learning Models
- Model Evaluation
- Visualization Techniques

to help businesses optimize logistics operations and improve customer satisfaction.

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

##  Project Structure

supply-chain-delivery-prediction/
│
├── supply_chain_dataset.csv
├── supply_chain_prediction.py
├── Project Overview.pdf
├── Explaination.pdf
└── README.md
 
---

##  Data Preprocessing

###  Data Cleaning
- Removed missing values
- Checked data consistency
- Validated feature quality

###  Encoding
Categorical features were converted into numerical values using:
- Label Encoding

Encoded Features:
- Shipping_Mode
- Order_Region

###  Feature Engineering
Created a new feature:
- Delay → Difference between actual and expected shipping days

---

##  Machine Learning Models

Two machine learning models were implemented:

###  Random Forest Classifier
- Handles complex relationships
- Provides feature importance visualization

###  Neural Network (MLPClassifier)
- Captures deeper non-linear patterns
- Uses feature scaling for optimization

---

##  Features Used

- Shipping_Mode
- Order_Region
- Order_Item_Quantity
- Product_Price
- Shipping_Days
- Delay

---

##  Model Training

- Dataset split:
  - 80% Training
  - 20% Testing

- Applied:
  - Feature Scaling
  - GridSearchCV for hyperparameter tuning

---

##  Model Evaluation

Models were evaluated using:

- Accuracy Score
- Mean Squared Error (MSE)
- Classification Report
- Confusion Matrix

Visualizations included:
- Feature Importance
- Confusion Matrices

---

##  Key Insights

- Delay was identified as the most influential feature affecting delivery risk.
- Random Forest provided better interpretability.
- Neural Networks captured more complex delivery patterns.

---

## Project Goals
. Predict shipment delays
. Improve logistics efficiency
. Reduce late delivery risks
. Support data-driven decision making
. Enhance customer satisfaction

##  How to Run

### 1. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
