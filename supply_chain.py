import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Load dataset
data = pd.read_csv("supply_chain_dataset.csv")
data = data.dropna()

# Features and target
features = [
    "Shipping_Mode",
    "Order_Region",
    "Order_Item_Quantity",
    "Product_Price",
    "Shipping_Days"
]
X = data[features]
y = data["Late_delivery_risk"]

# Encode categorical features
le = LabelEncoder()
for col in ["Shipping_Mode", "Order_Region"]:
    X[col] = le.fit_transform(X[col])

# ------------------------
# Add Delay feature
# ------------------------
# If Expected_Shipping_Days column exists:
if 'Expected_Shipping_Days' in data.columns:
    X['Delay'] = X['Shipping_Days'] - data['Expected_Shipping_Days']
else:
    # Otherwise, approximate using mean Shipping_Days per mode and region
    expected_days = X.groupby(['Shipping_Mode','Order_Region'])['Shipping_Days'].transform('mean')
    X['Delay'] = X['Shipping_Days'] - expected_days

# Only keep positive delays
X['Delay'] = X['Delay'].apply(lambda x: x if x > 0 else 0)

print("First 10 rows with Delay feature:")
print(X.head(10))

# Add Delay to features list
features.append('Delay')

# ------------------------
# Train/Test Split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# Feature Scaling for Neural Network
# ------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================
# Random Forest Model
# ========================
print("\n==============================")
print("Random Forest Model")
print("==============================")

# Grid Search for Random Forest
rf = RandomForestClassifier(random_state=42)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

rf_pred = rf_best.predict(X_test)

# Metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

print("Best Random Forest params:", rf_grid.best_params_)
print("Accuracy:", rf_accuracy)
print("Mean Squared Error:", rf_mse)
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))
print("Confusion Matrix:")
cm_rf = confusion_matrix(y_test, rf_pred)
print(cm_rf)

# Feature Importance Visualization
importance = rf_best.feature_importances_
plt.figure(figsize=(8,5))
sns.barplot(x=importance, y=features)
plt.title("Random Forest Feature Importance")

# Confusion Matrix Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ========================
# Neural Network Model
# ========================
print("\n==============================")
print("Neural Network Model")
print("==============================")

nn = MLPClassifier(max_iter=500, random_state=42)

# Grid Search for Neural Network
nn_params = {
    'hidden_layer_sizes': [(50,), (100,)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001]
}
nn_grid = GridSearchCV(nn, nn_params, cv=3, scoring='accuracy', n_jobs=-1)
nn_grid.fit(X_train_scaled, y_train)
nn_best = nn_grid.best_estimator_

nn_pred = nn_best.predict(X_test_scaled)

# Metrics
nn_accuracy = accuracy_score(y_test, nn_pred)
nn_mse = mean_squared_error(y_test, nn_pred)

print("Best Neural Network params:", nn_grid.best_params_)
print("Accuracy:", nn_accuracy)
print("Mean Squared Error:", nn_mse)
print("\nClassification Report:")
print(classification_report(y_test, nn_pred))
print("Confusion Matrix:")
cm_nn = confusion_matrix(y_test, nn_pred)
print(cm_nn)

# Confusion Matrix Heatmap for Neural Network
plt.figure(figsize=(5,4))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens')
plt.title("Neural Network Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()