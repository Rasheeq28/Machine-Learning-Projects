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

        # Load dataset from local directory
        mergedtrain = pd.read_csv(r"C:\Users\rashe\PycharmProjects\student_score_prediction\mergedtrain.csv")

        # Show preview
        st.write("#### Preview of Merged Train Data")
        st.dataframe(mergedtrain.head())

        st.markdown("### Merged Stores, Features and Test data")

        # --- EDA on numerical columns ---
        st.subheader("Exploratory Data Analysis (Numerical Columns)")
        num_cols = mergedtrain.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Summary stats
        st.write("#### Summary Statistics")
        st.dataframe(mergedtrain[num_cols].describe().T)

        # Correlation heatmap
        st.write("#### Correlation Heatmap")
        corr = mergedtrain[num_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap (Numerical Features)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Distribution plots for numerical features
        st.write("#### Distributions of Numerical Columns")
        for col in num_cols:
            fig = px.histogram(
                mergedtrain,
                x=col,
                nbins=30,
                marginal="box",
                title=f"Distribution of {col}"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Interactive scatter plots
        st.write("#### Scatter Plots (Numeric Relationships)")
        if "Weekly_Sales" in mergedtrain.columns:
            for col in [c for c in num_cols if c != "Weekly_Sales"]:
                fig = px.scatter(
                    mergedtrain,
                    x=col,
                    y="Weekly_Sales",
                    trendline="ols",
                    title=f"{col} vs Weekly Sales"
                )
                st.plotly_chart(fig, use_container_width=True)

    # ---------------- Testing ----------------
    with test_tab:
        st.subheader("🧪 Testing Phase")
        st.write("Here you can implement testing of models using hold-out / unseen datasets.")

    # ---------------- Training ----------------
    with train_tab:
        st.subheader("🏋️ Training Phase")
        st.write("Here you can implement model training pipelines (XGBoost, LightGBM, etc.).")

