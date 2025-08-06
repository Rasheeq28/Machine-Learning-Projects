# import kagglehub
#
# # Download latest version
# path = kagglehub.dataset_download("lainguyn123/student-performance-factors")
#
# print("Path to dataset files:", path)


# import pandas as pd
#
# # Load your dataset
# file_path = r"StudentPerformanceFactors.csv"
# df = pd.read_csv(file_path)
#
# missing_data = df[df['Exam_Score'].isna() | df['Hours_Studied'].isna()]
#
# # Display results
# print(missing_data)
#
# # If you want to save it to a CSV
# missing_data.to_csv("missing_exam_or_hours.csv", index=False)


# import pandas as pd
# df = pd.read_csv("StudentPerformanceFactors.csv")
# df = df[['Hours_Studied', 'Exam_Score']]
# print(df)

#
# import pandas as pd
#
# # Load dataset
# df = pd.read_csv(r"StudentPerformanceFactors.csv")
#
# # 1️⃣ Remove rows with missing/null values across ALL columns
# df_clean = df.dropna()
#
# # 2️⃣ Keep only 'Exam_Score' and 'Hours_Studied' columns
# df_final = df_clean[['Hours_Studied', 'Exam_Score']]
# df_final.to_csv("cleaned_hours_exam.csv", index=False)
# # Display the cleaned data
# print(df_final)


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
#
# # Load dataset
# df = pd.read_csv(r"StudentPerformanceFactors.csv")
#
# # Remove missing values and keep only required columns
# df_clean = df.dropna()[['Hours_Studied', 'Exam_Score']]
#
# # Split data
# X = df_clean[['Hours_Studied']]
# y = df_clean['Exam_Score']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Predictions
# y_pred = model.predict(X_test)
#
# # Scatter plot of actual data
# plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
#
# # Regression line
# x_line = np.linspace(X.min(), X.max(), 100)  # evenly spaced hours studied
# x_line_df = pd.DataFrame(x_line, columns=['Hours_Studied'])  # keep feature name
# y_line = model.predict(x_line_df)  # predicted scores for those hours
#
# plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')
#
#
# plt.title('Hours Studied vs Exam Score with Regression Line')
# plt.xlabel('Hours Studied (per week)')
# plt.ylabel('Exam Score')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()
#
# # Evaluation
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
#
#
# plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # perfect fit line
# plt.xlabel("Actual Exam Score")
# plt.ylabel("Predicted Exam Score")
# plt.title("Actual vs Predicted Exam Scores")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()
#
#
# plt.scatter(X_test, y_test, color='blue', label='Actual')
# plt.scatter(X_test, y_pred, color='red', alpha=0.6, label='Predicted')
# plt.xlabel('Hours Studied')
# plt.ylabel('Exam Score')
# plt.title('Actual vs Predicted Exam Scores')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()
#
# print(f"MAE: {mae:.2f}")
# print(f"MSE: {mse:.2f}")
# print(f"RMSE: {rmse:.2f}")
# print(f"R² Score: {r2:.2f}")
# print(f"Regression Equation: Exam_Score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Hours_Studied")


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
#
# # ---------------------------
# # 1️⃣ Load and Clean Dataset
# # ---------------------------
# df = pd.read_csv(r"StudentPerformanceFactors.csv")
# df_clean = df.dropna()[['Hours_Studied', 'Exam_Score']]
#
# # ---------------------------
# # 2️⃣ Split Data
# # ---------------------------
# X = df_clean[['Hours_Studied']]
# y = df_clean['Exam_Score']
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# # ---------------------------
# # 3️⃣ Train Model
# # ---------------------------
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
#
# # ---------------------------
# # 4️⃣ Plot 1: Regression Line on All Data
# # ---------------------------
# plt.figure()
# plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
#
# x_line = np.linspace(X.min(), X.max(), 100)
# x_line_df = pd.DataFrame(x_line, columns=['Hours_Studied'])
# y_line = model.predict(x_line_df)
#
# plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')
# plt.title('Hours Studied vs Exam Score with Regression Line')
# plt.xlabel('Hours Studied (per week)')
# plt.ylabel('Exam Score')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()
#
# # ---------------------------
# # 5️⃣ Plot 2: Predicted vs Actual (Perfect Fit Line)
# # ---------------------------
# plt.figure()
# plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
# plt.plot(
#     [y_test.min(), y_test.max()],
#     [y_test.min(), y_test.max()],
#     color='red',
#     linewidth=2
# )
# plt.xlabel("Actual Exam Score")
# plt.ylabel("Predicted Exam Score")
# plt.title("Actual vs Predicted Exam Scores")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()
#
# # ---------------------------
# # 6️⃣ Plot 3: Test Data (Actual vs Predicted)
# # ---------------------------
# plt.figure()
# plt.scatter(X_test, y_test, color='blue', label='Actual')
# plt.scatter(X_test, y_pred, color='red', alpha=0.6, label='Predicted')
# plt.xlabel('Hours Studied')
# plt.ylabel('Exam Score')
# plt.title('Actual vs Predicted Exam Scores')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()
#
# # ---------------------------
# # 7️⃣ Model Evaluation
# # ---------------------------
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
#
# print(f"MAE: {mae:.2f}")
# print(f"MSE: {mse:.2f}")
# print(f"RMSE: {rmse:.2f}")
# print(f"R² Score: {r2:.2f}")
# print(f"Regression Equation: Exam_Score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Hours_Studied")

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
#
# st.title("📊 Student Exam Score Predictor")
# st.write("Predict exam scores based on **Hours Studied** using Linear Regression.")
#
# # Upload CSV
# uploaded_file = st.file_uploader("StudentPerformanceFactors", type=["csv"])
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#
#     # Clean data
#     df_clean = df.dropna()[['Hours_Studied', 'Exam_Score']]
#
#     # Split data
#     X = df_clean[['Hours_Studied']]
#     y = df_clean['Exam_Score']
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#
#     # Train model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     # Plot 1: Regression Line
#     fig1, ax1 = plt.subplots()
#     ax1.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
#     x_line = np.linspace(X.min(), X.max(), 100)
#     x_line_df = pd.DataFrame(x_line, columns=['Hours_Studied'])
#     y_line = model.predict(x_line_df)
#     ax1.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')
#     ax1.set_title('Hours Studied vs Exam Score with Regression Line')
#     ax1.set_xlabel('Hours Studied (per week)')
#     ax1.set_ylabel('Exam Score')
#     ax1.legend()
#     st.pyplot(fig1)
#
#     # Plot 2: Predicted vs Actual (Perfect Fit Line)
#     fig2, ax2 = plt.subplots()
#     ax2.scatter(y_test, y_pred, color='blue', alpha=0.6)
#     ax2.plot([y_test.min(), y_test.max()],
#              [y_test.min(), y_test.max()],
#              color='red', linewidth=2)
#     ax2.set_xlabel("Actual Exam Score")
#     ax2.set_ylabel("Predicted Exam Score")
#     ax2.set_title("Actual vs Predicted Exam Scores")
#     st.pyplot(fig2)
#
#     # Plot 3: Test Data (Actual vs Predicted)
#     fig3, ax3 = plt.subplots()
#     ax3.scatter(X_test, y_test, color='blue', label='Actual')
#     ax3.scatter(X_test, y_pred, color='red', alpha=0.6, label='Predicted')
#     ax3.set_xlabel('Hours Studied')
#     ax3.set_ylabel('Exam Score')
#     ax3.set_title('Actual vs Predicted Exam Scores')
#     ax3.legend()
#     st.pyplot(fig3)
#
#     # Model Evaluation
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)
#
#     st.subheader("Model Evaluation Metrics")
#     st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
#     st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
#     st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
#     st.write(f"**R² Score:** {r2:.2f}")
#     st.write(f"**Regression Equation:** `Exam_Score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Hours_Studied`")
# else:
#     st.info("👆 Please upload your `StudentPerformanceFactors.csv` file to continue.")
#
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
#
# st.title("📊 Student Exam Score Predictor")
# st.write("Predict exam scores based on **Hours Studied** using Linear Regression.")
#
# # Load CSV from online URL
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df = pd.read_csv(csv_url)
#
# # Clean data
# df_clean = df.dropna()[['Hours_Studied', 'Exam_Score']]
#
# # Split data
# X = df_clean[['Hours_Studied']]
# y = df_clean['Exam_Score']
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# # Train model
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
#
# # Plot 1: Regression Line
# fig1, ax1 = plt.subplots()
# ax1.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
# x_line = np.linspace(X.min(), X.max(), 100)
# x_line_df = pd.DataFrame(x_line, columns=['Hours_Studied'])
# y_line = model.predict(x_line_df)
# ax1.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')
# ax1.set_title('Hours Studied vs Exam Score with Regression Line')
# ax1.set_xlabel('Hours Studied (per week)')
# ax1.set_ylabel('Exam Score')
# ax1.legend()
# st.pyplot(fig1)
#
# # Plot 2: Predicted vs Actual (Perfect Fit Line)
# fig2, ax2 = plt.subplots()
# ax2.scatter(y_test, y_pred, color='blue', alpha=0.6)
# ax2.plot([y_test.min(), y_test.max()],
#          [y_test.min(), y_test.max()],
#          color='red', linewidth=2)
# ax2.set_xlabel("Actual Exam Score")
# ax2.set_ylabel("Predicted Exam Score")
# ax2.set_title("Actual vs Predicted Exam Scores")
# st.pyplot(fig2)
#
# # Plot 3: Test Data (Actual vs Predicted)
# fig3, ax3 = plt.subplots()
# ax3.scatter(X_test, y_test, color='blue', label='Actual')
# ax3.scatter(X_test, y_pred, color='red', alpha=0.6, label='Predicted')
# ax3.set_xlabel('Hours Studied')
# ax3.set_ylabel('Exam Score')
# ax3.set_title('Actual vs Predicted Exam Scores')
# ax3.legend()
# st.pyplot(fig3)
#
# # Model Evaluation
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
#
# st.subheader("Model Evaluation Metrics")
# st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
# st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
# st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
# st.write(f"**R² Score:** {r2:.2f}")
# st.write(f"**Regression Equation:** `Exam_Score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Hours_Studied`")

#
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
#
# st.set_page_config(page_title="AI Prediction Dashboard", layout="wide")
#
# st.title("📊 AI Prediction Dashboard")
#
# # Create tabs
# tab1, tab2, tab3, tab4 = st.tabs([
#     "Student Score Predictor",
#     "Customer Segmentation",
#     "Loan Approval Prediction",
#     "Sales Forecasting"
# ])
#
# # ===================== TAB 1: Student Score Predictor =====================
# with tab1:
#     st.subheader("🎓 Student Exam Score Predictor")
#     st.write("Predict exam scores based on **Hours Studied** using Linear Regression.")
#
#     # Load CSV from URL
#     csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
#     df = pd.read_csv(csv_url)
#
#     # Clean data
#     df_clean = df.dropna()[['Hours_Studied', 'Exam_Score']]
#
#     # Split data
#     X = df_clean[['Hours_Studied']]
#     y = df_clean['Exam_Score']
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#
#     # Train model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     # Plot 1: Regression Line
#     fig1, ax1 = plt.subplots()
#     ax1.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
#     x_line = np.linspace(X.min(), X.max(), 100)
#     x_line_df = pd.DataFrame(x_line, columns=['Hours_Studied'])
#     y_line = model.predict(x_line_df)
#     ax1.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')
#     ax1.set_title('Hours Studied vs Exam Score with Regression Line')
#     ax1.set_xlabel('Hours Studied (per week)')
#     ax1.set_ylabel('Exam Score')
#     ax1.legend()
#     st.pyplot(fig1)
#
#     # Plot 2: Predicted vs Actual (Perfect Fit Line)
#     fig2, ax2 = plt.subplots()
#     ax2.scatter(y_test, y_pred, color='blue', alpha=0.6)
#     ax2.plot([y_test.min(), y_test.max()],
#              [y_test.min(), y_test.max()],
#              color='red', linewidth=2)
#     ax2.set_xlabel("Actual Exam Score")
#     ax2.set_ylabel("Predicted Exam Score")
#     ax2.set_title("Actual vs Predicted Exam Scores")
#     st.pyplot(fig2)
#
#     # Plot 3: Test Data (Actual vs Predicted)
#     fig3, ax3 = plt.subplots()
#     ax3.scatter(X_test, y_test, color='blue', label='Actual')
#     ax3.scatter(X_test, y_pred, color='red', alpha=0.6, label='Predicted')
#     ax3.set_xlabel('Hours Studied')
#     ax3.set_ylabel('Exam Score')
#     ax3.set_title('Actual vs Predicted Exam Scores')
#     ax3.legend()
#     st.pyplot(fig3)
#
#     # Model Evaluation
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)
#
#     st.subheader("📈 Model Evaluation Metrics")
#     st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
#     st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
#     st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
#     st.write(f"**R² Score:** {r2:.2f}")
#     st.write(f"**Regression Equation:** `Exam_Score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Hours_Studied`")
#
# # ===================== TAB 2: Customer Segmentation =====================
# with tab2:
#     st.subheader("🛍 Customer Segmentation")
#     st.info("This section will display customer segmentation insights. (Coming soon!)")
#
# # ===================== TAB 3: Loan Approval Prediction =====================
# with tab3:
#     st.subheader("🏦 Loan Approval Prediction")
#     st.info("This section will display loan approval predictions. (Coming soon!)")
#
# # ===================== TAB 4: Sales Forecasting =====================
# with tab4:
#     st.subheader("📈 Sales Forecasting")
#     st.info("This section will display sales forecasting results. (Coming soon!)")


#
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="AI Prediction Dashboard", layout="wide")
st.title("📊 AI Prediction Dashboard")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Student Score Predictor",
    "Customer Segmentation",
    "Loan Approval Prediction",
    "Sales Forecasting"
])

# ================================= TAB 1 =================================
with tab1:
    subtab1, subtab2, subtab3 = st.tabs(["Model & Visualizations", "Whole Dataset", "Test Data"])

    # Load CSV from GitHub raw link (replace with your actual link)
    csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
    df = pd.read_csv(csv_url)

    # ---------- Sub-tab 1: Model & Visualizations ----------
    with subtab1:
        st.subheader("🎓 Student Exam Score Predictor")
        st.write("Predict exam scores based on **Hours Studied** using Linear Regression.")

        # Dataset cleaning
        df_clean = df.dropna()[['Hours_Studied', 'Exam_Score']]

        # Display row counts
        st.write(f"**📄 Actual Dataset Row Count:** {len(df)}")
        st.write(f"**🧹 Cleaned Dataset Row Count:** {len(df_clean)}")

        # Split data
        X = df_clean[['Hours_Studied']]
        y = df_clean['Exam_Score']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Plot 1: Regression Line
        fig1, ax1 = plt.subplots()
        ax1.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
        x_line = np.linspace(X.min(), X.max(), 100)
        x_line_df = pd.DataFrame(x_line, columns=['Hours_Studied'])
        y_line = model.predict(x_line_df)
        ax1.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')
        ax1.set_title('Hours Studied vs Exam Score with Regression Line')
        ax1.set_xlabel('Hours Studied (per week)')
        ax1.set_ylabel('Exam Score')
        ax1.legend()
        st.pyplot(fig1)

        # Plot 2: Predicted vs Actual
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_test, y_pred, color='blue', alpha=0.6)
        ax2.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 color='red', linewidth=2)
        ax2.set_xlabel("Actual Exam Score")
        ax2.set_ylabel("Predicted Exam Score")
        ax2.set_title("Actual vs Predicted Exam Scores")
        st.pyplot(fig2)

        # Plot 3: Test Data (Actual vs Predicted)
        fig3, ax3 = plt.subplots()
        ax3.scatter(X_test, y_test, color='blue', label='Actual')
        ax3.scatter(X_test, y_pred, color='red', alpha=0.6, label='Predicted')
        ax3.set_xlabel('Hours Studied')
        ax3.set_ylabel('Exam Score')
        ax3.set_title('Actual vs Predicted Exam Scores')
        ax3.legend()
        st.pyplot(fig3)

        # Model Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📈 Model Evaluation Metrics")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")
        st.write(f"**Regression Equation:** `Exam_Score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Hours_Studied`")

    # ---------- Sub-tab 2: Whole Dataset ----------
    with subtab2:
        st.subheader("📂 Whole Dataset")
        st.dataframe(df, use_container_width=True)

    # ---------- Sub-tab 3: Test Data ----------
    with subtab3:
        st.subheader("🧪 Test Dataset (20% Split)")
        test_df = X_test.copy()
        test_df['Actual_Exam_Score'] = y_test.values
        st.dataframe(test_df, use_container_width=True)






