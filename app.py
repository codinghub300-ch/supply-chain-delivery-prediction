import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Supply Chain Delivery Prediction",
    layout="wide"
)

st.title("📦 Supply Chain Delivery Prediction Dashboard")
st.markdown(
"""
Predict whether an order is likely to be delivered late using:

- Shipping Mode
- Region
- Quantity
- Product Price
- Shipping Days
"""
)

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv("supply_chain_dataset.csv")
df = df.dropna()

# ==========================================
# ENCODING
# ==========================================
X = df[[
    "Shipping_Mode",
    "Order_Region",
    "Order_Item_Quantity",
    "Product_Price",
    "Shipping_Days"
]]

y = df["Late_delivery_risk"]

shipping_encoder = LabelEncoder()
region_encoder = LabelEncoder()

X["Shipping_Mode"] = shipping_encoder.fit_transform(X["Shipping_Mode"])
X["Order_Region"] = region_encoder.fit_transform(X["Order_Region"])

# ==========================================
# DELAY FEATURE
# ==========================================
expected_days = X.groupby(
    ["Shipping_Mode", "Order_Region"]
)["Shipping_Days"].transform("mean")

X["Delay"] = X["Shipping_Days"] - expected_days
X["Delay"] = X["Delay"].apply(lambda x: x if x > 0 else 0)

features = X.columns

# ==========================================
# SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# RANDOM FOREST
# ==========================================
rf = RandomForestClassifier(random_state=42)

rf_params = {
    "n_estimators":[100,200],
    "max_depth":[5,10,None]
}

rf_grid = GridSearchCV(
    rf,
    rf_params,
    cv=3
)

rf_grid.fit(X_train, y_train)

rf_model = rf_grid.best_estimator_

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

# ==========================================
# NEURAL NETWORK
# ==========================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn = MLPClassifier(max_iter=500)

nn_params = {
    "hidden_layer_sizes":[(50,), (100,)],
    "activation":["relu","tanh"]
}

nn_grid = GridSearchCV(
    nn,
    nn_params,
    cv=3
)

nn_grid.fit(X_train_scaled, y_train)

nn_model = nn_grid.best_estimator_

nn_pred = nn_model.predict(X_test_scaled)

nn_accuracy = accuracy_score(y_test, nn_pred)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Prediction Inputs")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Random Forest", "Neural Network"]
)

shipping_mode = st.sidebar.selectbox(
    "Shipping Mode",
    df["Shipping_Mode"].unique()
)

region = st.sidebar.selectbox(
    "Order Region",
    df["Order_Region"].unique()
)

quantity = st.sidebar.number_input(
    "Quantity",
    min_value=1,
    value=5
)

price = st.sidebar.number_input(
    "Product Price",
    min_value=1,
    value=100
)

days = st.sidebar.number_input(
    "Shipping Days",
    min_value=1,
    value=4
)

# ==========================================
# PREDICTION
# ==========================================
if st.sidebar.button("Predict"):

    ship_encoded = shipping_encoder.transform([shipping_mode])[0]
    region_encoded = region_encoder.transform([region])[0]

    delay = max(
        days - X["Shipping_Days"].mean(),
        0
    )

    sample = [[
        ship_encoded,
        region_encoded,
        quantity,
        price,
        days,
        delay
    ]]

    if model_choice == "Random Forest":
        prediction = rf_model.predict(sample)[0]

    else:
        sample = scaler.transform(sample)
        prediction = nn_model.predict(sample)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠ High Risk of Late Delivery")
    else:
        st.success("✅ Delivery On Time")

# ==========================================
# ACCURACY
# ==========================================
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Random Forest Accuracy",
        f"{rf_accuracy*100:.2f}%"
    )

with col2:
    st.metric(
        "Neural Network Accuracy",
        f"{nn_accuracy*100:.2f}%"
    )

# ==========================================
# CONFUSION MATRIX
# ==========================================
st.markdown("---")
st.subheader("Random Forest Confusion Matrix")

cm = confusion_matrix(y_test, rf_pred)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

st.pyplot(fig)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
st.subheader("Feature Importance")

importance = rf_model.feature_importances_

fig2, ax2 = plt.subplots(figsize=(7,4))

sns.barplot(
    x=importance,
    y=features,
    ax=ax2
)

st.pyplot(fig2)

# ==========================================
# DATASET
# ==========================================
st.markdown("---")

st.subheader("Dataset")

st.dataframe(
    df,
    use_container_width=True
)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================
st.markdown("---")

st.subheader("Classification Report")

report = classification_report(
    y_test,
    rf_pred,
    output_dict=True
)

st.dataframe(
    pd.DataFrame(report).transpose()
)
