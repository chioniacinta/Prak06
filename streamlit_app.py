import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # To balance the classes

def main():
    st.title("Diabetes Prediction Web App")
    st.sidebar.title("Diabetes Prediction Web App")
    st.markdown("Predict whether a person has diabetes using machine learning models 🩺")
    st.sidebar.markdown("Predict whether a person has diabetes using machine learning models 🩺")

    @st.cache_data
    def load_data():
        data = pd.read_csv('diabetes.csv')  # Ensure 'diabetes.csv' file is in the same directory
        return data

    @st.cache_data
    def split(df):
        y = df['Outcome']  # Target column
        x = df.drop(columns=['Outcome'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        
        # Balance the dataset using SMOTE (Synthetic Minority Over-sampling Technique)
        smote = SMOTE(random_state=0)
        x_train, y_train = smote.fit_resample(x_train, y_train)
        
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list, model, x_test, y_test):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            disp = ConfusionMatrixDisplay.from_estimator(
                model, x_test, y_test,
                display_labels=class_names,
                cmap=plt.cm.Blues,
                normalize=None
            )
            st.pyplot(disp.figure_)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            disp = RocCurveDisplay.from_estimator(
                model, x_test, y_test,
                name='ROC Curve',
                color='darkorange'
            )
            st.pyplot(disp.figure_)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            disp = PrecisionRecallDisplay.from_estimator(
                model, x_test, y_test,
                name='Precision-Recall Curve',
                color='green'
            )
            st.pyplot(disp.figure_)

    df = load_data()
    class_names = ['No Diabetes', 'Diabetes']

    x_train, x_test, y_train, y_test = split(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    # Feature scaling (important for SVM and Logistic Regression)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = model.score(x_test, y_test)
            
            # Metrics
            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))
            st.write("F1 Score:", round(f1_score(y_test, y_pred), 2))
            st.write("ROC AUC Score:", round(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]), 2))
            plot_metrics(metrics, model, x_test, y_test)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = model.score(x_test, y_test)

            # Metrics
            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))
            st.write("F1 Score:", round(f1_score(y_test, y_pred), 2))
            st.write("ROC AUC Score:", round(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]), 2))
            plot_metrics(metrics, model, x_test, y_test)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                bootstrap=(bootstrap == 'True'),
                n_jobs=-1
            )
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = model.score(x_test, y_test)

            # Metrics
            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))
            st.write("F1 Score:", round(f1_score(y_test, y_pred), 2))
            st.write("ROC AUC Score:", round(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]), 2))
            plot_metrics(metrics, model, x_test, y_test)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Pima Indians Diabetes Dataset (Classification)")
        st.write(df)

if __name__ == '__main__':
    import matplotlib.pyplot as plt  # Make sure matplotlib is imported after scikit-learn
    main()