# project all
from sklearn.cluster import DBSCAN, KMeans
import plotly.express as px
import warnings
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold, StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import IPython.display as display
from IPython.display import display
import xgboost as xgb
import missingno as msno
import os

st.set_page_config(page_title="Machine Learning projects", layout="wide")
st.title("📊 Machine Learning projects")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Student Score Predictor",
    "Customer Segmentation",
    "Loan Approval Prediction",
    "Sales Forecasting"
])

# ================================= TAB 1 =================================

with tab1:
    st.subheader("📊 Student Score Predictor")
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load dataset
    csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
    df_raw = pd.read_csv(csv_url)
    df = df_raw.copy()

    # --- Advanced Missing Value Imputation ---
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    target = "Exam_Score"

    features = [
        "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
        "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
        "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
        "Family_Income", "Teacher_Quality", "School_Type",
        "Peer_Influence",
        "Physical_Activity",
        "Learning_Disabilities", "Parental_Education_Level",
        "Distance_from_Home"
         "Gender"
    ]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df[target]

    # --- Preprocessing Pipelines ---
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    poly_features_list = ["Hours_Studied", "Previous_Scores", "Sleep_Hours"]

    numeric_poly_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    numeric_scaler = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor_poly = ColumnTransformer(
        transformers=[
            ('poly_num', numeric_poly_transformer, poly_features_list),
            ('other_num', numeric_scaler, [col for col in numeric_cols if col not in poly_features_list]),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='passthrough'
    )

    preprocessor_linear = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), numeric_cols),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough'
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Hyperparameter Tuning ---
    st.subheader("Hyperparameter Tuning with GridSearchCV")

    param_grid = {'regressor__alpha': np.logspace(-4, 4, 10)}

    multi_linear_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_linear),
        ('regressor', Ridge())
    ])

    poly_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_poly),
        ('regressor', Ridge())
    ])

    cv_strategy = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    tuned_multi_linear = GridSearchCV(multi_linear_pipeline, param_grid, cv=cv_strategy, scoring='r2', verbose=1)
    tuned_poly = GridSearchCV(poly_pipeline, param_grid, cv=cv_strategy, scoring='r2', verbose=1)

    tuned_multi_linear.fit(X, y)
    tuned_poly.fit(X, y)

    best_linear_params = tuned_multi_linear.best_params_
    best_linear_score = tuned_multi_linear.best_score_
    best_poly_params = tuned_poly.best_params_
    best_poly_score = tuned_poly.best_score_

    st.write(f"Best Alpha for Multi-Feature Linear Regression (Ridge): **{best_linear_params['regressor__alpha']:.4f}** (R²: {best_linear_score:.4f})")
    st.write(f"Best Alpha for Multi-Feature Polynomial (deg=2, Ridge): **{best_poly_params['regressor__alpha']:.4f}** (R²: {best_poly_score:.4f})")

    # Define models
    models = {
        "Simple Linear Regression": Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        "Multi-Feature Linear Regression (Tuned Ridge)": tuned_multi_linear.best_estimator_,
        "Multi-Feature Polynomial (Tuned Ridge)": tuned_poly.best_estimator_
    }

    # --- Cross-Validation ---
    st.subheader("Model Performance with Cross-Validation")
    cv_results = []
    r2_scores_dict = {}

    for name, model in models.items():
        if name == "Simple Linear Regression":
            r2_scores = cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='r2')
            mae_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_absolute_error')
            mse_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(mse_scores)
        else:
            r2_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2')
            mae_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_absolute_error')
            mse_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(mse_scores)

        r2_scores_dict[name] = r2_scores
        cv_results.append([
            name,
            np.mean(r2_scores),
            np.std(r2_scores),
            np.mean(mae_scores),
            np.mean(mse_scores),
            np.mean(rmse_scores)
        ])

    cv_df = pd.DataFrame(cv_results,
                         columns=["Model", "Mean CV R²", "Std Dev CV R²", "Mean CV MAE", "Mean CV MSE", "Mean CV RMSE"])
    st.dataframe(cv_df.set_index("Model"))

    # --- Visualization of CV Results ---
    st.subheader("R² Scores for Each Cross-Validation Fold")
    fig_cv = go.Figure()
    for name, scores in r2_scores_dict.items():
        fig_cv.add_trace(go.Box(y=scores, name=name))

    fig_cv.update_layout(title="R² Scores Distribution across 15 Folds",
                         yaxis_title="R² Score",
                         showlegend=False)
    st.plotly_chart(fig_cv, use_container_width=True)

    # --- Actual vs Predicted ---
    st.subheader("Actual vs Predicted Scores on Test Set")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))

    predictions = {}
    for name, model in models.items():
        if name == "Simple Linear Regression":
            X_train_specific = X_train[["Hours_Studied"]]
            X_test_specific = X_test[["Hours_Studied"]]
        else:
            X_train_specific = X_train
            X_test_specific = X_test

        model.fit(X_train_specific, y_train)
        y_pred = model.predict(X_test_specific)
        predictions[name] = y_pred
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))

    fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
    st.plotly_chart(fig, use_container_width=True)

    # --- Test Set Metrics ---
    st.subheader("Test Set Performance Metrics")

    test_metrics = []
    for name, y_pred in predictions.items():
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        test_metrics.append([name, r2, mae, mse, rmse])

    test_metrics_df = pd.DataFrame(test_metrics,
                                   columns=["Model", "R²", "MAE", "MSE", "RMSE"])
    st.dataframe(test_metrics_df.set_index("Model"))


# ========================== customer segmentation ==========================
with tab2:
    st.subheader("🧍 Customer Segmentation using Clustering")

    # Create subtabs
    data_tab, viz_tab = st.tabs(["📂 Dataset & Preprocessing", "📊 Visualizations & Interpretations"])

    with data_tab:
        # Load dataset from URL
        df_customers = pd.read_csv("https://raw.githubusercontent.com/Rasheeq28/datasets/refs/heads/main/Mall_Customers.csv")

        st.write("### Raw Dataset Preview")
        st.dataframe(df_customers.head(), use_container_width=True)

        # Select relevant features: Annual Income and Spending Score
        df_selected = df_customers[['Annual Income (k$)', 'Spending Score (1-100)']]

        # Standardize features
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_selected)

        st.write("### Scaled Features Preview")
        st.dataframe(pd.DataFrame(df_scaled, columns=['Annual Income (scaled)', 'Spending Score (scaled)']), use_container_width=True)

    with viz_tab:
        # Elbow method to find optimal number of clusters
        inertia = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(df_scaled)
            inertia.append(kmeans.inertia_)

        fig_elbow = px.line(
            x=list(k_range),
            y=inertia,
            labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'},
            title="🔍 Elbow Method for Optimal K"
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

        st.write("""
        **Interpretation:**
        The elbow plot shows the inertia decreasing as K increases. The 'elbow' point suggests the optimal number of clusters.
        Here, K=5 is chosen to balance complexity and fit.
        """)

        # KMeans clustering with optimal_k=5
        optimal_k = 5
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df_customers['Cluster'] = kmeans.fit_predict(df_scaled)

        fig_cluster = px.scatter(
            df_customers,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            color=df_customers['Cluster'].astype(str),
            title="💠 Customer Clusters (KMeans)",
            labels={'Cluster': 'Segment'},
            template="plotly"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.write("""
        **Cluster Insights:**
        - Colors represent distinct segments based on income and spending.
        - Use to tailor marketing: e.g., premium offers for high-income/high-spenders.
        """)

        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.6, min_samples=5)
        df_customers['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)

        fig_dbscan = px.scatter(
            df_customers,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            color=df_customers['DBSCAN_Cluster'].astype(str),
            title="🔷 DBSCAN Clustering Results",
            labels={'DBSCAN_Cluster': 'Segment'},
            template="plotly_dark"
        )
        st.plotly_chart(fig_dbscan, use_container_width=True)

        st.write("""
        **DBSCAN Analysis:**
        - Detects clusters by density and highlights outliers (-1).
        - Outliers could be niche customers or anomalies.
        """)

        # Cluster averages summary (KMeans)
        st.write("### 🧾 Average Spending per Cluster (KMeans)")
        cluster_summary = df_customers.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
        st.dataframe(cluster_summary)

        st.write("""
        **Cluster Averages Interpretation:**
        - Shows average income & spending per cluster.
        - Identify high-value segments for focused engagement.
        """)

        # Cluster sizes
        st.write("### 📊 Cluster Sizes")
        cluster_counts = df_customers['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)

        st.write("""
        **Cluster Size Insights:**
        - Larger clusters mean bigger customer groups; smaller ones are niches.
        - Allocate marketing resources accordingly.
        """)

        st.markdown("---")
        st.write("🔄 **Note:** Features were standardized to give equal importance before clustering.")


with tab3:
    st.subheader("🏦 Loan Approval Prediction")

    dataset_tab, model_tab, dtree_tab, explanation_tab = st.tabs([
        "📂 Dataset", "📈 Logistic Regression Model", "🌳 Decision Tree Model", "📝 Explanation"
    ])

    # --- Dataset Subtab ---
    with dataset_tab:
        st.write("### Raw Dataset Preview")
        url = "https://raw.githubusercontent.com/Rasheeq28/datasets/refs/heads/main/loan_approval_dataset.csv"
        df_raw = pd.read_csv(url)
        st.dataframe(df_raw, use_container_width=True)

    # --- Logistic Regression Model Subtab ---
    with model_tab:
        # Load and preprocess dataset
        url = "https://raw.githubusercontent.com/Rasheeq28/datasets/refs/heads/main/loan_approval_dataset.csv"
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        df["loan_status"] = df["loan_status"].str.strip()
        df["Debt_Income"] = df["loan_amount"] / df["income_annum"]

        y = df["loan_status"].map({"Approved": 1, "Rejected": 0})
        df = df[~y.isna()]
        y = y.dropna()

        X = df.drop(columns=["loan_id", "loan_status"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        # Logistic Regression pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        lr_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42))
        ])

        # 5-Fold CV for Logistic Regression
        cv_folds = 5
        lr_cv_scores = cross_val_score(lr_pipeline, X, y, cv=cv_folds, scoring="accuracy")
        st.markdown(f"### 5-Fold Cross-Validation Accuracy")
        st.write(f"Logistic Regression CV Accuracy: {lr_cv_scores.mean():.3f} ± {lr_cv_scores.std():.3f}")

        # Fit on train and evaluate on test
        lr_pipeline.fit(X_train, y_train)
        y_pred = lr_pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        st.markdown(f"### Model Accuracy on Test Set: **{accuracy * 100:.4f}%**")

        st.markdown("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

        # Confusion matrix with Plotly heatmap (interactive)
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Rejected", "Approved"]

        total = cm.sum()
        percentages = cm / total * 100

        z_text = [[f"{count}<br>{percent:.1f}%" for count, percent in zip(row_counts, row_percs)]
                  for row_counts, row_percs in zip(cm, percentages)]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=z_text,
            texttemplate="%{text}",
            colorscale='Blues',
            hoverongaps=False,
            colorbar=dict(title="Count")
        ))

        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            yaxis_autorange='reversed',
            width=600,
            height=500,
            template="plotly_white"
        )

        st.plotly_chart(fig_cm, use_container_width=True)

        # Interpretation text for confusion matrix
        st.markdown("""
        **Confusion Matrix Interpretation:**
        - **True Positives (Top-Left):** Number of loans correctly predicted as Approved.
        - **True Negatives (Bottom-Right):** Number of loans correctly predicted as Rejected.
        - **False Positives (Top-Right):** Loans predicted as Approved but were Rejected (Type I error).
        - **False Negatives (Bottom-Left):** Loans predicted as Rejected but were Approved (Type II error).

        Ideally, we want to maximize true positives and true negatives while minimizing false positives and false negatives.
        """)

        # Accuracy bar chart (Plotly)
        fig_acc = go.Figure(data=go.Bar(
            x=["Accuracy"],
            y=[accuracy],
            marker_color='green',
            text=[f"{accuracy * 100:.4f}%"],
            textposition='auto'
        ))

        fig_acc.update_layout(
            yaxis=dict(range=[0, 1]),
            title="Model Accuracy",
            template="plotly_white",
            width=400,
            height=400
        )

        st.plotly_chart(fig_acc, use_container_width=False)
    #
    # --- Decision Tree Model Subtab ---
    with dtree_tab:
        url = "https://raw.githubusercontent.com/Rasheeq28/datasets/refs/heads/main/loan_approval_dataset.csv"
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        df["loan_status"] = df["loan_status"].str.strip()
        df["Debt_Income"] = df["loan_amount"] / df["income_annum"]

        y = df["loan_status"].map({"Approved": 1, "Rejected": 0})
        df = df[~y.isna()]
        y = y.dropna()

        X = df.drop(columns=["loan_id", "loan_status"])

        # Check class balance
        st.write("### Class distribution")
        st.write(Counter(y))

        # Use stratified split to maintain class proportions in train/test
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in strat_split.split(X, y):
            X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

        # Fit preprocessors only on train data
        numeric_transformer = StandardScaler().fit(X_train_raw[numeric_features])
        categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore").fit(
            X_train_raw[categorical_features])

        # Transform train data
        X_train_num = numeric_transformer.transform(X_train_raw[numeric_features])
        X_train_cat = categorical_transformer.transform(X_train_raw[categorical_features]).toarray()
        import numpy as np

        X_train = np.hstack([X_train_num, X_train_cat])

        # Transform test data using train-fit transformers
        X_test_num = numeric_transformer.transform(X_test_raw[numeric_features])
        X_test_cat = categorical_transformer.transform(X_test_raw[categorical_features]).toarray()
        X_test = np.hstack([X_test_num, X_test_cat])

        # Initialize and train Decision Tree
        dtree = DecisionTreeClassifier(random_state=42)
        dtree.fit(X_train, y_train)

        # Predict on test
        y_pred = dtree.predict(X_test)

        # Calculate and display multiple metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.markdown(f"### Decision Tree Model Metrics on Test Set")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1-score: {f1:.4f}")

        # Confusion matrix with Plotly heatmap
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Rejected", "Approved"]
        total = cm.sum()
        percentages = cm / total * 100

        z_text = [[f"{count}<br>{percent:.1f}%" for count, percent in zip(row_counts, row_percs)]
                  for row_counts, row_percs in zip(cm, percentages)]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=z_text,
            texttemplate="%{text}",
            colorscale='Blues',
            hoverongaps=False,
            colorbar=dict(title="Count")
        ))

        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            yaxis_autorange='reversed',
            width=600,
            height=500,
            template="plotly_white"
        )

        st.plotly_chart(fig_cm, use_container_width=True)

        # Accuracy bar chart
        fig_acc = go.Figure(data=go.Bar(
            x=["Accuracy", "Precision", "Recall", "F1"],
            y=[accuracy, precision, recall, f1],
            marker_color=['green', 'blue', 'orange', 'purple'],
            text=[f"{v * 100:.2f}%" for v in [accuracy, precision, recall, f1]],
            textposition='auto'
        ))

        fig_acc.update_layout(
            yaxis=dict(range=[0, 1]),
            title="Model Metrics",
            template="plotly_white",
            width=600,
            height=400
        )

        st.plotly_chart(fig_acc, use_container_width=False)

        st.markdown("""
        **Important Notes:**

        - Preprocessing (scaling, encoding) is done only on train data, then applied to test data to avoid data leakage.
        - Stratified split maintains class proportions in train and test sets.
        - Evaluate multiple metrics (accuracy, precision, recall, F1) for balanced insight.
        - Check class distribution above — if very imbalanced, consider methods like SMOTE or class weights.
        - Inspect features manually to ensure no direct leakage of target information.
        """)
with tab4:
    st.subheader("📈 Walmart Sales Forecasting")

    # Create subtabs
    eda_tab, test_tab, train_tab = st.tabs(["📊 EDA and Merging", "🧪 Testing", "🏋️ Training"])

    # ---------------- EDA and Merging ----------------
    with eda_tab:
        st.markdown("### Merged Stores, Features and Train, Cleaned data and encoded categorical columns")

    with test_tab:
        st.subheader("🧪 Testing Phase")

        # --- Create subtabs for different store types ---
        type_a_tab, type_b_tab, type_c_tab = st.tabs(["Type A Stores", "Type B Stores", "Type C Stores"])

        # ---------------- Type A Stores ----------------
        with type_a_tab:
            st.markdown("### Testing for Type A Stores")


            # --- 0. Load dataset with caching ---
            @st.cache_data
            def load_data():
                file_id = "1GgRiYJe3rpXwojA75qeRuMNy-ZobpoC1"
                download_url = f"https://drive.google.com/uc?id={file_id}"
                df = pd.read_csv(download_url)
                df["Date"] = pd.to_datetime(df["Date"])
                return df


            merged = load_data()

            # Filter for Type A stores if needed
            # merged_a = merged[merged["StoreType"] == "A"].copy()

            # --- 1. Split train and test ---
            train_df = merged[merged["Date"] < "2012-11-02"].copy()
            test_df = merged[merged["Date"] >= "2012-11-02"].copy()

            features = [col for col in merged.columns if col not in ["Weekly_Sales", "Date"]]
            X_train = train_df[features]
            y_train = train_df["Weekly_Sales"]
            X_test = test_df[features]


            # --- 2. Train XGBoost model with caching ---
            @st.cache_resource
            def train_model(X, y):
                model = XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y)
                return model


            with st.spinner("Training XGBoost model for Type A Stores..."):
                model = train_model(X_train, y_train)
            st.success("Model training complete!")

            # --- 3. Predict Weekly_Sales for test_df ---
            test_df = test_df.copy()
            test_df["Weekly_Sales_Predicted"] = model.predict(X_test)

            # --- 4. Aggregate predicted weekly sales per Date ---
            predicted_table = test_df.groupby("Date")["Weekly_Sales_Predicted"].sum().reset_index()
            predicted_table = predicted_table.sort_values("Date").reset_index(drop=True)

            st.markdown("### 📅 Predicted Weekly Sales per Date")
            st.dataframe(predicted_table)

            # --- 5. Evaluate if actuals exist ---
            if "Weekly_Sales" in test_df.columns and test_df["Weekly_Sales"].notna().any():
                y_true = test_df["Weekly_Sales"]
                y_pred = test_df["Weekly_Sales_Predicted"]

                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                st.markdown("### 📊 Model Performance on Test Set")
                st.write(f"**RMSE:** {rmse:.2f}")
                st.write(f"**MAE:** {mae:.2f}")
                st.write(f"**R²:** {r2:.4f}")

            # --- 6. Visualization ---
            st.markdown("### 📈 Actual vs Predicted Weekly Sales")

            train_agg = train_df.groupby("Date")["Weekly_Sales"].sum().reset_index()
            fig = go.Figure()

            # Actual train sales
            fig.add_trace(go.Scatter(
                x=train_agg["Date"],
                y=train_agg["Weekly_Sales"],
                mode='lines+markers',
                name='Actual (Train)',
                line=dict(color='blue'),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Sales: %{y}<extra></extra>'
            ))

            # Predicted test sales
            fig.add_trace(go.Scatter(
                x=predicted_table["Date"],
                y=predicted_table["Weekly_Sales_Predicted"],
                mode='lines+markers',
                name='Predicted (Test)',
                line=dict(color='orange'),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Predicted Sales: %{y}<extra></extra>'
            ))

            fig.update_layout(
                title="Actual vs Predicted Weekly Sales (Type A Stores)",
                xaxis_title="Date",
                yaxis_title="Weekly Sales",
                template="plotly_white",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)


        with type_b_tab:
            st.markdown("### Testing for Type B Stores")


            # --- 0. Load dataset with caching ---
            @st.cache_data
            def load_data_b():
                file_id = "1fqENpAd8SPY9tq-pXAf4W_QSDtDAxaHf"
                download_url = f"https://drive.google.com/uc?id={file_id}"
                df = pd.read_csv(download_url)
                df["Date"] = pd.to_datetime(df["Date"])
                return df


            mergedB = load_data_b()

            # Filter for Type B stores
            # mergedB = mergedB[mergedB["StoreType"] == "B"].copy()

            # --- 1. Split train and test ---
            train_df = mergedB[mergedB["Date"] < "2012-11-02"].copy()
            test_df = mergedB[mergedB["Date"] >= "2012-11-02"].copy()

            features = [col for col in mergedB.columns if col not in ["Weekly_Sales", "Date"]]
            X_train = train_df[features]
            y_train = train_df["Weekly_Sales"]
            X_test = test_df[features]


            # --- 2. Train XGBoost model with caching ---
            @st.cache_resource
            def train_model_b(X, y):
                model = XGBRegressor(
                    n_estimators=3000,
                    learning_rate=0.045,
                    max_depth=6,
                    subsample=1,
                    colsample_bytree=0.85,
                    min_child_weight=5,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y)
                return model


            with st.spinner("Training XGBoost model for Type B Stores..."):
                model = train_model_b(X_train, y_train)
            st.success("Model training complete!")

            # --- 3. Predict Weekly_Sales for test_df ---
            test_df = test_df.copy()
            test_df["Weekly_Sales_Predicted"] = model.predict(X_test)

            # --- 4. Aggregate predicted weekly sales per Date ---
            predicted_table = test_df.groupby("Date")["Weekly_Sales_Predicted"].sum().reset_index()
            predicted_table = predicted_table.sort_values("Date").reset_index(drop=True)

            st.markdown("### 📅 Predicted Weekly Sales per Date")
            st.dataframe(predicted_table)

            # --- 5. Evaluate if actuals exist ---
            if "Weekly_Sales" in test_df.columns and test_df["Weekly_Sales"].notna().any():
                y_true = test_df["Weekly_Sales"]
                y_pred = test_df["Weekly_Sales_Predicted"]

                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                st.markdown("### 📊 Model Performance on Test Set")
                st.write(f"**RMSE:** {rmse:.2f}, **MAE:** {mae:.2f}, **R²:** {r2:.4f}")

            # --- 6. Visualization ---
            train_agg = train_df.groupby("Date")["Weekly_Sales"].sum().reset_index()

            fig = go.Figure()

            # Actual train sales
            fig.add_trace(go.Scatter(
                x=train_agg["Date"],
                y=train_agg["Weekly_Sales"],
                mode='lines+markers',
                name='Actual (Train)',
                line=dict(color='blue'),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Sales: %{y}<extra></extra>'
            ))

            # Predicted test sales
            fig.add_trace(go.Scatter(
                x=predicted_table["Date"],
                y=predicted_table["Weekly_Sales_Predicted"],
                mode='lines+markers',
                name='Predicted (Test)',
                line=dict(color='orange'),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Sales: %{y}<extra></extra>'
            ))

            fig.update_layout(
                title="Weekly Sales: Actual (Train) vs Predicted (Test) (Type B Stores)",
                xaxis_title="Date",
                yaxis_title="Weekly Sales (Aggregated)",
                hovermode="x unified",
                template="plotly_white",
                width=900,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        with type_c_tab:
            st.markdown("### Testing for Type C Stores")


            # --- 0. Load dataset with caching ---
            @st.cache_data
            def load_data_c():
                file_id = "1LRO5Vybi0gHjpLwvyOYG_Vwp6k03pf6I"
                download_url = f"https://drive.google.com/uc?id={file_id}"
                df = pd.read_csv(download_url)
                df["Date"] = pd.to_datetime(df["Date"])
                return df


            mergedC = load_data_c()

            # --- 1. Split train and test ---
            train_df = mergedC[mergedC["Date"] < "2012-11-02"].copy()
            test_df = mergedC[mergedC["Date"] >= "2012-11-02"].copy()

            features = [col for col in mergedC.columns if col not in ["Weekly_Sales", "Date"]]
            X_train = train_df[features]
            y_train = train_df["Weekly_Sales"]
            X_test = test_df[features]


            # --- 2. Train XGBoost model with caching ---
            @st.cache_resource
            def train_model_c(X, y):
                model = XGBRegressor(
                    n_estimators=1500,
                    learning_rate=0.01,
                    max_depth=6,
                    subsample=1,
                    colsample_bytree=0.85,
                    min_child_weight=1,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y)
                return model


            with st.spinner("Training XGBoost model for Type C Stores..."):
                model = train_model_c(X_train, y_train)
            st.success("Model training complete!")

            # --- 3. Predict Weekly_Sales for test_df ---
            test_df = test_df.copy()
            test_df["Weekly_Sales_Predicted"] = model.predict(X_test)

            # --- 4. Aggregate predicted weekly sales per Date ---
            predicted_table = test_df.groupby("Date")["Weekly_Sales_Predicted"].sum().reset_index()
            predicted_table = predicted_table.sort_values("Date").reset_index(drop=True)

            st.markdown("### 📅 Predicted Weekly Sales per Date")
            st.dataframe(predicted_table)

            # --- 5. Evaluate if actuals exist ---
            if "Weekly_Sales" in test_df.columns and test_df["Weekly_Sales"].notna().any():
                y_true = test_df["Weekly_Sales"]
                y_pred = test_df["Weekly_Sales_Predicted"]

                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                st.markdown("### 📊 Model Performance on Test Set")
                st.write(f"**RMSE:** {rmse:.2f}, **MAE:** {mae:.2f}, **R²:** {r2:.4f}")

            # --- 6. Visualization ---
            train_agg = train_df.groupby("Date")["Weekly_Sales"].sum().reset_index()

            fig = go.Figure()

            # Actual train sales
            fig.add_trace(go.Scatter(
                x=train_agg["Date"],
                y=train_agg["Weekly_Sales"],
                mode='lines+markers',
                name='Actual (Train)',
                line=dict(color='blue'),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Sales: %{y}<extra></extra>'
            ))

            # Predicted test sales
            fig.add_trace(go.Scatter(
                x=predicted_table["Date"],
                y=predicted_table["Weekly_Sales_Predicted"],
                mode='lines+markers',
                name='Predicted (Test)',
                line=dict(color='orange'),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Sales: %{y}<extra></extra>'
            ))

            fig.update_layout(
                title="Weekly Sales: Actual (Train) vs Predicted (Test) (Type C Stores)",
                xaxis_title="Date",
                yaxis_title="Weekly Sales (Aggregated)",
                hovermode="x unified",
                template="plotly_white",
                width=900,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

    # ---------------- Training ----------------
    with train_tab:
        st.markdown("## 🏋️ Model Training")

        type_a_train_tab, type_b_train_tab, type_c_train_tab = st.tabs(
            ["Type A Training", "Type B Training", "Type C Training"]
        )

        # ==============================
        # TYPE A TRAINING
        # ==============================
        with type_a_train_tab:
            st.markdown("### 📊 Training - Type A Stores")


            # --- 0. Load A_train with caching ---
            @st.cache_data
            def load_a_train():
                file_id = "1zTMNyl48gpiUd2njuMZk81OW-Ux4aYq3"
                download_url = f"https://drive.google.com/uc?id={file_id}"
                df = pd.read_csv(download_url)
                df["Date"] = pd.to_datetime(df["Date"])
                return df


            A_train = load_a_train()

            # --- Step 1: Feature Engineering ---
            merged = A_train.copy()

            # Days since start
            merged["DaysSinceStart"] = (merged["Date"] - merged["Date"].min()).dt.days

            # Lag features (Weekly_Sales)
            merged["lag1"] = merged.groupby("Store")["Weekly_Sales"].shift(1)
            merged["lag2"] = merged.groupby("Store")["Weekly_Sales"].shift(2)
            merged["lag4"] = merged.groupby("Store")["Weekly_Sales"].shift(4)
            merged["lag52"] = merged.groupby("Store")["Weekly_Sales"].shift(52)  # same week last year

            # Rolling averages
            merged["rolling4"] = merged.groupby("Store")["Weekly_Sales"].shift(1).rolling(4).mean()
            merged["rolling8"] = merged.groupby("Store")["Weekly_Sales"].shift(1).rolling(8).mean()

            # Holiday lag features
            merged["holidaylag1"] = merged.groupby("Store")["IsHoliday"].shift(1)
            merged["holidaylag2"] = merged.groupby("Store")["IsHoliday"].shift(2)
            merged["holidaylag8"] = merged.groupby("Store")["IsHoliday"].shift(8)

            # Week-before-holiday flag
            merged["WeekbeforeHoliday"] = (
                merged.groupby("Store")["IsHoliday"].shift(-1).fillna(0).astype(int)
            )

            # Drop NA rows
            merged = merged.dropna().reset_index(drop=True)

            # --- Step 2: Train/Test Split ---
            split_idx = int(len(merged["Date"].unique()) * 0.8)
            train_dates = merged["Date"].unique()[:split_idx]
            test_dates = merged["Date"].unique()[split_idx:]

            train = merged[merged["Date"].isin(train_dates)]
            test = merged[merged["Date"].isin(test_dates)]

            X_train = train.drop(columns=["Weekly_Sales", "Date"])
            y_train = train["Weekly_Sales"]
            X_test = test.drop(columns=["Weekly_Sales", "Date"])
            y_test = test["Weekly_Sales"]


            # --- Step 3: Train Model (cached) ---
            @st.cache_resource
            def train_model_a(X, y):
                model = XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y)
                return model


            with st.spinner("Training XGBoost model for Type A Stores..."):
                model = train_model_a(X_train, y_train)
            st.success("✅ Model training complete for Type A Stores!")

            # --- Step 4: Predict & Evaluate ---
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.markdown("### 📈 Model Performance")
            st.write(f"**RMSE:** {rmse:.2f}")
            st.write(f"**MAE:** {mae:.2f}")
            st.write(f"**R²:** {r2:.4f}")

            # --- Step 5: Aggregated Visualization ---
            agg_plot = test.copy()
            agg_plot["Predicted"] = y_pred
            agg_plot = agg_plot.groupby("Date")[["Weekly_Sales", "Predicted"]].sum().reset_index()

            import plotly.graph_objects as go

            fig = go.Figure()

            # Actual sales
            fig.add_trace(go.Scatter(
                x=agg_plot["Date"],
                y=agg_plot["Weekly_Sales"],
                mode="lines+markers",
                name="Actual",
                line=dict(color="blue"),
                marker=dict(size=6),
                hovertemplate="Date: %{x}<br>Sales: %{y}<extra></extra>"
            ))

            # Predicted sales
            fig.add_trace(go.Scatter(
                x=agg_plot["Date"],
                y=agg_plot["Predicted"],
                mode="lines+markers",
                name="Predicted",
                line=dict(color="orange"),
                marker=dict(size=6),
                hovertemplate="Date: %{x}<br>Sales: %{y}<extra></extra>"
            ))

            fig.update_layout(
                title="Actual vs Predicted Weekly Sales (Type A Stores)",
                xaxis_title="Date",
                yaxis_title="Weekly Sales",
                hovermode="x unified",
                template="plotly_white",
                width=900,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
        # ==============================
        # TYPE B TRAINING
        # ==============================
        # ==============================
        # TYPE B TRAINING (with Holiday & Lag Features)
        # ==============================
        with type_b_train_tab:
            st.markdown("### 📊 Training - Type B Stores (with Holiday & Lag Features)")


            # --- 0. Load B_train with caching ---
            @st.cache_data
            def load_b_train():
                file_id = "10LeqAxi2fuU_rbV4rQWZlOpUUFlMK8Xn"
                download_url = f"https://drive.google.com/uc?id={file_id}"
                df = pd.read_csv(download_url)
                df["Date"] = pd.to_datetime(df["Date"])
                return df


            B_train = load_b_train()

            # --- Step 1: Preprocessing ---
            df = B_train.copy()


            # Holiday classification
            def get_holiday_type(date):
                date = pd.to_datetime(date)
                if date.month == 12 and date.day == 25:
                    return "Christmas"
                if date.month == 11 and date.weekday() == 3 and 22 <= date.day <= 28:
                    return "Thanksgiving"
                if date.month == 9 and date.weekday() == 0 and date.day <= 7:
                    return "Labor Day"
                if date.month == 2 and date.weekday() == 6 and date.day <= 7:
                    return "SuperBowl"
                return "None"


            df["HolidayType"] = df["Date"].apply(get_holiday_type)
            df = pd.get_dummies(df, columns=["HolidayType"], drop_first=True)

            # Sort for lag features
            df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

            # Regular lag & rolling features
            lag_list = [1, 2, 4, 8]
            rolling_windows = [2, 4, 8]

            for lag in lag_list:
                df[f"lag_{lag}"] = df.groupby("Store")["Weekly_Sales"].shift(lag)

            for window in rolling_windows:
                df[f"rolling_{window}"] = (
                    df.groupby("Store")["Weekly_Sales"].shift(1).rolling(window).mean()
                )

            # Holiday-specific lag & rolling
            holiday_lags = [1, 2, 4, 8, 12, 16]
            holiday_windows = [2, 4, 8, 12]

            for lag in holiday_lags:
                df[f"holiday_lag_{lag}"] = (
                        df["IsHoliday"] * df.groupby("Store")["Weekly_Sales"].shift(lag)
                )

            for window in holiday_windows:
                df[f"holiday_rolling_{window}"] = (
                        df["IsHoliday"] *
                        df.groupby("Store")["Weekly_Sales"].shift(1).rolling(window).mean()
                )

            # Exponential weighted moving averages
            df["holiday_ewm_8"] = (
                    df["IsHoliday"] *
                    df.groupby("Store")["Weekly_Sales"].shift(1).ewm(span=8, adjust=False).mean()
            )
            df["holiday_ewm_16"] = (
                    df["IsHoliday"] *
                    df.groupby("Store")["Weekly_Sales"].shift(1).ewm(span=16, adjust=False).mean()
            )

            # Previous holiday week effect
            for holiday in ["Christmas", "Thanksgiving", "Labor Day", "SuperBowl"]:
                col_name = f"HolidayType_{holiday}"
                if col_name in df.columns:
                    df[f"last_{holiday}"] = (
                            df.groupby("Store")["Weekly_Sales"].shift(52) * df[col_name]
                    )
                else:
                    df[f"last_{holiday}"] = 0

            # Drop NA from shifting
            df = df.dropna().reset_index(drop=True)

            # --- Step 2: Train/Test Split ---
            split_idx = int(len(df["Date"].unique()) * 0.8)
            train_dates = df["Date"].unique()[:split_idx]
            test_dates = df["Date"].unique()[split_idx:]

            train = df[df["Date"].isin(train_dates)]
            test = df[df["Date"].isin(test_dates)]

            X_train = train.drop(columns=["Weekly_Sales", "Date"])
            y_train = train["Weekly_Sales"]
            X_test = test.drop(columns=["Weekly_Sales", "Date"])
            y_test = test["Weekly_Sales"]

            # --- Step 3: Outlier Handling ---
            y_train = np.clip(y_train, y_train.quantile(0.01), y_train.quantile(0.99))


            # --- Step 4: Train Model (cached) ---
            @st.cache_resource
            def train_model_b(X, y, X_val, y_val):
                model = XGBRegressor(
                    n_estimators=3000,
                    learning_rate=0.045,
                    max_depth=6,
                    subsample=1,
                    colsample_bytree=0.85,
                    min_child_weight=5,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
                return model


            with st.spinner("Training XGBoost model for Type B Stores..."):
                model = train_model_b(X_train, y_train, X_test, y_test)
            st.success("✅ Model training complete for Type B Stores!")

            # --- Step 5: Predict & Evaluate ---
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.markdown("### 📈 Model Performance")
            st.write(f"**RMSE:** {rmse:.2f}")
            st.write(f"**MAE:** {mae:.2f}")
            st.write(f"**R²:** {r2:.4f}")

            # --- Step 6: Aggregated Visualization ---
            agg_plot = test.copy()
            agg_plot["Predicted"] = y_pred
            agg_plot = agg_plot.groupby("Date")[["Weekly_Sales", "Predicted"]].sum().reset_index()

            import plotly.graph_objects as go

            fig = go.Figure()

            # Actual sales
            fig.add_trace(go.Scatter(
                x=agg_plot["Date"],
                y=agg_plot["Weekly_Sales"],
                mode="lines+markers",
                name="Actual",
                line=dict(color="blue"),
                marker=dict(size=6),
                hovertemplate="Date: %{x}<br>Sales: %{y}<extra></extra>"
            ))

            # Predicted sales
            fig.add_trace(go.Scatter(
                x=agg_plot["Date"],
                y=agg_plot["Predicted"],
                mode="lines+markers",
                name="Predicted",
                line=dict(color="orange"),
                marker=dict(size=6),
                hovertemplate="Date: %{x}<br>Sales: %{y}<extra></extra>"
            ))

            fig.update_layout(
                title="Actual vs Predicted Weekly Sales (Type B Stores with Holiday & Lag Features)",
                xaxis_title="Date",
                yaxis_title="Weekly Sales",
                hovermode="x unified",
                template="plotly_white",
                width=900,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        # ==============================
        # TYPE C TRAINING (Simplified, No Lags/Rollings)
        # ==============================
        with type_c_train_tab:
            st.markdown("### 📊 Training - Type C Stores (Simplified Features)")


            # --- Step 0: Load C_train ---
            @st.cache_data
            def load_c_train():
                file_id = "1TPms7rHBbe6s7GmnoIiKC18GxS8wU6QZ"
                download_url = f"https://drive.google.com/uc?id={file_id}"
                df = pd.read_csv(download_url)
                df["Date"] = pd.to_datetime(df["Date"])
                return df


            C_train = load_c_train()

            # --- Step 1: Preprocessing ---
            df = C_train.copy()


            # Drop NAs if any (from shift)
            df = df.dropna().reset_index(drop=True)

            # --- Step 2: Train/Test Split ---
            split_idx = int(len(df["Date"].unique()) * 0.8)
            train_dates = df["Date"].unique()[:split_idx]
            test_dates = df["Date"].unique()[split_idx:]

            train = df[df["Date"].isin(train_dates)]
            test = df[df["Date"].isin(test_dates)]

            X_train = train.drop(columns=["Weekly_Sales", "Date"])
            y_train = train["Weekly_Sales"]
            X_test = test.drop(columns=["Weekly_Sales", "Date"])
            y_test = test["Weekly_Sales"]


            # --- Step 3: Train Model ---
            @st.cache_resource
            def train_model_c(X, y):
                model = XGBRegressor(
                    n_estimators=3000,
                    learning_rate=0.045,
                    max_depth=6,
                    subsample=1,
                    colsample_bytree=0.85,
                    min_child_weight=5,

                )
                model.fit(X, y)
                return model


            with st.spinner("Training XGBoost model for Type C Stores..."):
                model = train_model_c(X_train, y_train)
            st.success("✅ Model training complete for Type C Stores!")

            # --- Step 4: Predict & Evaluate ---
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.markdown("### 📈 Model Performance")
            st.write(f"**RMSE:** {rmse:.2f}")
            st.write(f"**MAE:** {mae:.2f}")
            st.write(f"**R²:** {r2:.4f}")

            # --- Step 5: Aggregated Visualization ---
            agg_plot = test.copy()
            agg_plot["Predicted"] = y_pred
            agg_plot = agg_plot.groupby("Date")[["Weekly_Sales", "Predicted"]].sum().reset_index()

            import plotly.graph_objects as go

            fig = go.Figure()

            # Actual sales
            fig.add_trace(go.Scatter(
                x=agg_plot["Date"],
                y=agg_plot["Weekly_Sales"],
                mode="lines+markers",
                name="Actual",
                line=dict(color="blue"),
                marker=dict(size=6),
                hovertemplate="Date: %{x}<br>Sales: %{y}<extra></extra>"
            ))

            # Predicted sales
            fig.add_trace(go.Scatter(
                x=agg_plot["Date"],
                y=agg_plot["Predicted"],
                mode="lines+markers",
                name="Predicted",
                line=dict(color="orange"),
                marker=dict(size=6),
                hovertemplate="Date: %{x}<br>Sales: %{y}<extra></extra>"
            ))

            fig.update_layout(
                title="Actual vs Predicted Weekly Sales (Type C Stores - Simplified Features)",
                xaxis_title="Date",
                yaxis_title="Weekly Sales",
                hovermode="x unified",
                template="plotly_white",
                width=900,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
