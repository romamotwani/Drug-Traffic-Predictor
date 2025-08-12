# Save this file as app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


# --------- DATA LOADING & PREPROCESSING ---------
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\Purvesh.S.Mavle\\OneDrive\\Desktop\\python\\drug.csv")
    df['Flight_Time'] = np.random.randint(30, 600, size=len(df))

    # Encode Gender in the main dataframe itself
    if df['Gender'].dtype == 'object':
        df['Gender'] = pd.Categorical(df['Gender']).codes

    feature_columns = [
        'Drug Quantity (kg)', 'Age', 'Gender', 'Flight_Time',
        'Source Country_Brazil', 'Source Country_Colombia', 'Source Country_India',
        'Source Country_Mexico', 'Source Country_Netherlands', 'Source Country_Nigeria',
        'Source Country_Peru', 'Source Country_Thailand', 'Source Country_Venezuela',
        'Destination Country_Canada', 'Destination Country_China', 'Destination Country_France',
        'Destination Country_Germany', 'Destination Country_Italy', 'Destination Country_Japan',
        'Destination Country_Spain', 'Destination Country_UK', 'Destination Country_USA',
        'Flight Name_AirIndia', 'Flight Name_AmericanAir', 'Flight Name_BritishAir',
        'Flight Name_Delta', 'Flight Name_Emirates', 'Flight Name_KLM',
        'Flight Name_Lufthansa', 'Flight Name_Qantas', 'Flight Name_SingaporeAir'
    ]

    # Now build features X
    X = df[feature_columns].copy()
    X = X.fillna(0)

    # Create target y
    df['Trafficking'] = df[[col for col in df.columns if "Drug Name_" in col]].sum(axis=1) > 0
    y = df['Trafficking'].astype(int)

    return df, X, y

df, X, y = load_data()

# --------- TRAIN TEST SPLIT ---------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# --------- MODEL TRAINING ---------
@st.cache_resource
def train_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    # Hyperparameter Tuning
    param_grid_dt = {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'criterion': ['gini', 'entropy'],
    }
    grid_dt = GridSearchCV(models["Decision Tree"], param_grid_dt, cv=5, n_jobs=-1, scoring='accuracy')
    grid_dt.fit(X_train_balanced, y_train_balanced)
    models["Decision Tree"] = grid_dt.best_estimator_

    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    grid_rf = GridSearchCV(models["Random Forest"], param_grid_rf, cv=5, n_jobs=-1, scoring='accuracy')
    grid_rf.fit(X_train_balanced, y_train_balanced)
    models["Random Forest"] = grid_rf.best_estimator_

    param_grid_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    grid_gb = GridSearchCV(models["Gradient Boosting"], param_grid_gb, cv=5, n_jobs=-1, scoring='accuracy')
    grid_gb.fit(X_train_balanced, y_train_balanced)
    models["Gradient Boosting"] = grid_gb.best_estimator_

    return models

models = train_models()

# --------- STREAMLIT UI ---------
st.title("🚀 Drug Trafficking Detection Dashboard")

model_choice = st.selectbox("Select a Model", list(models.keys()))

model = models[model_choice]

# Predict
if model_choice == "Logistic Regression":
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    X_input = X_test_scaled
else:
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    X_input = X_test

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.metric(label="Model Accuracy", value=f"{accuracy*100:.2f}%")

# --------- CONFUSION MATRIX ---------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# --------- NORMALIZED CONFUSION MATRIX ---------
st.subheader("Normalized Confusion Matrix")
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
fig2, ax2 = plt.subplots()
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
st.pyplot(fig2)

# --------- ROC CURVE ---------
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_input)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    st.subheader(f"ROC Curve (AUC = {roc_auc:.2f})")
    fig3, ax3 = plt.subplots()
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    ax3.plot([0, 1], [0, 1], 'k--')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend(loc="lower right")
    st.pyplot(fig3)

    # --------- PRECISION RECALL CURVE ---------
    precision, recall, _ = precision_recall_curve(y_test, probs)
    st.subheader("Precision-Recall Curve")
    fig4, ax4 = plt.subplots()
    ax4.plot(recall, precision, color='purple')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    st.pyplot(fig4)

# --------- FEATURE IMPORTANCE (if available) ---------
if model_choice in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
    st.subheader("Top 10 Feature Importances")
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    fig5, ax5 = plt.subplots()
    sns.barplot(data=importances, x='Importance', y='Feature', palette='viridis', ax=ax5)
    st.pyplot(fig5)
