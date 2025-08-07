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


#linear
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
#
# st.set_page_config(page_title="Machine Learning projects", layout="wide")
# st.title("📊 Machine Learning projects")
#
# # Main Tabs
# tab1, tab2, tab3, tab4 = st.tabs([
#     "Student Score Predictor",
#     "Customer Segmentation",
#     "Loan Approval Prediction",
#     "Sales Forecasting"
# ])
#
# # ================================= TAB 1 =================================
# with tab1:
#     subtab1, subtab2, subtab3 = st.tabs(["Model & Visualizations", "Whole Dataset", "Test Data"])
#
#     # Load CSV from GitHub raw link (replace with your actual link)
#     csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
#     df = pd.read_csv(csv_url)
#
#     # ---------- Sub-tab 1: Model & Visualizations ----------
#     # ---------- Sub-tab 1: Model & Visualizations ----------
#     with subtab1:
#         st.subheader("🎓 Student Exam Score Predictor")
#         st.write("Predict exam scores based on **Hours Studied** using Linear Regression.")
#
#         # Dataset cleaning
#         df_clean = df.dropna()[['Hours_Studied', 'Exam_Score']]
#
#         # Display row counts
#         st.write(f"**📄 Actual Dataset Row Count:** {len(df)}")
#         st.write(f"**🧹 Cleaned Dataset Row Count:** {len(df_clean)}")
#
#         # Split data
#         X = df_clean[['Hours_Studied']]
#         y = df_clean['Exam_Score']
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )
#
#         # Train model
#         model = LinearRegression()
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#
#         # --- Arrange first two plots side-by-side ---
#         col1, col2 = st.columns(2)
#
#         # Plot 1: Hours Studied vs Exam Score (with Regression Line)
#         with col1:
#             x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
#             y_line = model.predict(x_line)
#             fig1 = go.Figure()
#             fig1.add_trace(go.Scatter(
#                 x=df_clean['Hours_Studied'], y=df_clean['Exam_Score'],
#                 mode='markers', name='Actual Data',
#                 marker=dict(color='blue')
#             ))
#             fig1.add_trace(go.Scatter(
#                 x=x_line.flatten(), y=y_line,
#                 mode='lines', name='Regression Line',
#                 line=dict(color='red')
#             ))
#             fig1.update_layout(
#                 title='📘 Hours Studied vs Exam Score',
#                 xaxis_title='Hours Studied (per week)',
#                 yaxis_title='Exam Score',
#                 legend=dict(orientation="h", y=-0.2)
#             )
#             st.plotly_chart(fig1, use_container_width=True)
#
#         # Plot 2: Actual vs Predicted Exam Scores
#         with col2:
#             fig2 = px.scatter(
#                 x=y_test, y=y_pred,
#                 labels={'x': 'Actual Exam Score', 'y': 'Predicted Exam Score'},
#                 title="🎯 Actual vs Predicted Exam Scores",
#             )
#             fig2.add_trace(go.Scatter(
#                 x=[y_test.min(), y_test.max()],
#                 y=[y_test.min(), y_test.max()],
#                 mode='lines',
#                 name='Perfect Prediction Line',
#                 line=dict(color='red')
#             ))
#             st.plotly_chart(fig2, use_container_width=True)
#
#         # Plot 3: Test Data Comparison
#         fig3 = go.Figure()
#         fig3.add_trace(go.Scatter(
#             x=X_test['Hours_Studied'], y=y_test,
#             mode='markers', name='Actual',
#             marker=dict(color='blue')
#         ))
#         fig3.add_trace(go.Scatter(
#             x=X_test['Hours_Studied'], y=y_pred,
#             mode='markers', name='Predicted',
#             marker=dict(color='red')
#         ))
#         fig3.update_layout(
#             title="🔍 Test Data: Actual vs Predicted",
#             xaxis_title='Hours Studied',
#             yaxis_title='Exam Score',
#             legend=dict(orientation="h", y=-0.2)
#         )
#         st.plotly_chart(fig3, use_container_width=True)
#         with subtab2:
#             st.subheader("📂 Full Dataset (Raw CSV)")
#             if df.empty:
#                 st.warning("Dataset is empty or not loaded correctly.")
#             else:
#                 st.dataframe(df, use_container_width=True)
#
#         # ------------------- Subtab 3: Test Data -------------------
#         with subtab3:
#             st.subheader("🧪 Test Data with Predictions")
#             test_data = X_test.copy()
#             test_data['Actual'] = y_test.values
#             test_data['Predicted'] = y_pred
#             st.dataframe(test_data.reset_index(drop=True), use_container_width=True)
#
#         # Model Evaluation
#         mae = mean_absolute_error(y_test, y_pred)
#         mse = mean_squared_error(y_test, y_pred)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_test, y_pred)
#
#         st.subheader("📈 Model Evaluation Metrics")
#         st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
#         st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
#         st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
#         st.write(f"**R² Score:** {r2:.2f}")
#         st.write(
#             f"**Regression Equation:** `Exam_Score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Hours_Studied`")
#

#
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
#
# st.set_page_config(page_title="Machine Learning projects", layout="wide")
# st.title("📊 Machine Learning projects")
#
# # Main Tabs
# tab1, tab2, tab3, tab4 = st.tabs([
#     "Student Score Predictor",
#     "Customer Segmentation",
#     "Loan Approval Prediction",
#     "Sales Forecasting"
# ])
#
# # ================================= TAB 1 =================================
# with tab1:
#     subtab_main, subtab2, subtab3 = st.tabs(["Model & Visualizations", "Whole Dataset", "Test Data"])
#
#     # Load dataset
#     csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
#     df = pd.read_csv(csv_url)
#
#     # Clean data
#     df_clean = df.dropna()[['Hours_Studied', 'Exam_Score']]
#     X = df_clean[['Hours_Studied']]
#     y = df_clean['Exam_Score']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Sub-tabs: Linear | Polynomial
#     with subtab_main:
#         model_tab1, model_tab2 = st.tabs(["Linear", "Polynomial"])
#
#         # ========================== LINEAR ==========================
#         with model_tab1:
#             st.subheader("🔵 Linear Regression Model")
#
#             # Train linear model
#             linear_model = LinearRegression()
#             linear_model.fit(X_train, y_train)
#             y_pred_linear = linear_model.predict(X_test)
#
#             # Plot 1: Regression Line
#             x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
#             y_line = linear_model.predict(x_line)
#             fig1 = go.Figure()
#             fig1.add_trace(go.Scatter(x=df_clean['Hours_Studied'], y=df_clean['Exam_Score'],
#                                       mode='markers', name='Actual Data', marker=dict(color='blue')))
#             fig1.add_trace(go.Scatter(x=x_line.flatten(), y=y_line,
#                                       mode='lines', name='Regression Line', line=dict(color='red')))
#             fig1.update_layout(title='📘 Hours Studied vs Exam Score (Linear)',
#                                xaxis_title='Hours Studied', yaxis_title='Exam Score')
#             st.plotly_chart(fig1, use_container_width=True)
#
#             # Plot 2: Actual vs Predicted
#             fig2 = px.scatter(x=y_test, y=y_pred_linear,
#                               labels={'x': 'Actual', 'y': 'Predicted'},
#                               title="🎯 Actual vs Predicted (Linear)")
#             fig2.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
#                                       y=[y_test.min(), y_test.max()],
#                                       mode='lines', name='Perfect Fit', line=dict(color='red')))
#             st.plotly_chart(fig2, use_container_width=True)
#
#             # Plot 3: Test Data
#             fig3 = go.Figure()
#             fig3.add_trace(go.Scatter(x=X_test['Hours_Studied'], y=y_test,
#                                       mode='markers', name='Actual', marker=dict(color='blue')))
#             fig3.add_trace(go.Scatter(x=X_test['Hours_Studied'], y=y_pred_linear,
#                                       mode='markers', name='Predicted', marker=dict(color='red')))
#             fig3.update_layout(title="🔍 Test Data: Actual vs Predicted (Linear)",
#                                xaxis_title='Hours Studied', yaxis_title='Exam Score')
#             st.plotly_chart(fig3, use_container_width=True)
#
#             # Evaluation
#             st.subheader("📈 Linear Regression Metrics")
#             st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_linear):.2f}")
#             st.write(f"**MSE:** {mean_squared_error(y_test, y_pred_linear):.2f}")
#             st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred_linear)):.2f}")
#             st.write(f"**R² Score:** {r2_score(y_test, y_pred_linear):.2f}")
#             st.write(f"**Equation:** `Exam_Score = {linear_model.intercept_:.2f} + {linear_model.coef_[0]:.2f} * Hours_Studied`")
#
#             st.subheader("📝 Predict Exam Score for Custom Hours Studied (Linear)")
#             hours_input = st.number_input("Enter hours studied:", min_value=0.0, max_value=100.0, value=5.0, step=0.1,
#                                           key="linear_input")
#             input_array = np.array([[hours_input]])
#             linear_pred = linear_model.predict(input_array)[0]
#             st.write(f"**Linear Regression Prediction:** {linear_pred:.2f} exam score")
#
#         # ========================== POLYNOMIAL ==========================
#         with model_tab2:
#             st.subheader("🟣 Polynomial Regression Model (Degree 2)")
#
#             # Polynomial model
#             poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
#             poly_model.fit(X_train, y_train)
#             y_pred_poly = poly_model.predict(X_test)
#
#             # Plot 1: Polynomial Curve
#             x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
#             y_line = poly_model.predict(x_line)
#             fig4 = go.Figure()
#             fig4.add_trace(go.Scatter(x=df_clean['Hours_Studied'], y=df_clean['Exam_Score'],
#                                       mode='markers', name='Actual Data', marker=dict(color='blue')))
#             fig4.add_trace(go.Scatter(x=x_line.flatten(), y=y_line,
#                                       mode='lines', name='Polynomial Curve', line=dict(color='green')))
#             fig4.update_layout(title='📗 Hours Studied vs Exam Score (Polynomial)',
#                                xaxis_title='Hours Studied', yaxis_title='Exam Score')
#             st.plotly_chart(fig4, use_container_width=True)
#
#             # Plot 2: Actual vs Predicted
#             fig5 = px.scatter(x=y_test, y=y_pred_poly,
#                               labels={'x': 'Actual', 'y': 'Predicted'},
#                               title="🎯 Actual vs Predicted (Polynomial)")
#             fig5.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
#                                       y=[y_test.min(), y_test.max()],
#                                       mode='lines', name='Perfect Fit', line=dict(color='red')))
#             st.plotly_chart(fig5, use_container_width=True)
#
#             # Plot 3: Test Data
#             fig6 = go.Figure()
#             fig6.add_trace(go.Scatter(x=X_test['Hours_Studied'], y=y_test,
#                                       mode='markers', name='Actual', marker=dict(color='blue')))
#             fig6.add_trace(go.Scatter(x=X_test['Hours_Studied'], y=y_pred_poly,
#                                       mode='markers', name='Predicted', marker=dict(color='green')))
#             fig6.update_layout(title="🔍 Test Data: Actual vs Predicted (Polynomial)",
#                                xaxis_title='Hours Studied', yaxis_title='Exam Score')
#             st.plotly_chart(fig6, use_container_width=True)
#
#             # Evaluation
#             st.subheader("📈 Polynomial Regression Metrics")
#             st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_poly):.2f}")
#             st.write(f"**MSE:** {mean_squared_error(y_test, y_pred_poly):.2f}")
#             st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred_poly)):.2f}")
#             st.write(f"**R² Score:** {r2_score(y_test, y_pred_poly):.2f}")
#             st.write("**Note:** Polynomial equation is inferred from model pipeline, not directly printed.")
#
#             st.subheader("📝 Predict Exam Score for Custom Hours Studied")
#             # Reuse the same input or add a new one (optional)
#             hours_input_poly = st.number_input("Enter hours studied:", min_value=0.0, max_value=100.0, value=5.0,
#                                                step=0.1, key="poly_input")
#             input_array_poly = np.array([[hours_input_poly]])
#             poly_pred = poly_model.predict(input_array_poly)[0]
#             st.write(f"**Polynomial Regression Prediction (Degree 2):** {poly_pred:.2f} exam score")
#
#     # ========================== WHOLE DATASET TAB ==========================
#     with subtab2:
#         st.subheader("📂 Full Dataset (Raw CSV)")
#         if df.empty:
#             st.warning("Dataset is empty or not loaded correctly.")
#         else:
#             st.dataframe(df, use_container_width=True)
#
#     # ========================== TEST DATA TAB ==========================
#     with subtab3:
#         st.subheader("🧪 Test Data with Predictions (Linear)")
#         test_data = X_test.copy()
#         test_data['Actual'] = y_test.values
#         test_data['Predicted (Linear)'] = y_pred_linear
#         test_data['Predicted (Poly)'] = y_pred_poly
#         st.dataframe(test_data.reset_index(drop=True), use_container_width=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# st.set_page_config(page_title="Student Score Predictor (Multi-feature)", layout="wide")
# st.title("📊 Student Exam Score Predictor with Multiple Features")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df = pd.read_csv(csv_url)
#
# # Select features to use
# numeric_features = [
#     'Hours_Studied', 'Attendance', 'Previous_Scores',
#     'Sleep_Hours', 'Tutoring_Sessions'
# ]
# categorical_features = [
#     'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level',
#     'Teacher_Quality', 'School_Type', 'Peer_Influence',
#     'Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities',
#     'Parental_Education_Level', 'Distance_from_Home', 'Gender'
# ]
#
# # Drop rows with missing target or features
# df = df.dropna(subset=['Exam_Score'] + numeric_features + categorical_features)
#
# # Prepare X and y
# X = df[numeric_features + categorical_features]
# y = df['Exam_Score']
#
# # Preprocessing pipelines
# numeric_transformer = 'passthrough'  # numeric features used as is
#
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])
#
# # Create a pipeline that preprocesses data then fits linear regression
# model = Pipeline(steps=[('preprocessor', preprocessor),
#                         ('regressor', LinearRegression())])
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Fit model
# model.fit(X_train, y_train)
#
# # Predict on test
# y_pred = model.predict(X_test)
#
# # Show metrics
# st.subheader("📈 Model Performance Metrics")
# st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
# st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
# st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
# st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
#
# st.markdown("---")
# st.header("📝 Predict Exam Score Based on Inputs")
#
# # Helper: Interpretations of features
# feature_descriptions = {
#     'Hours_Studied': "Number of hours spent studying per week. More hours generally improve scores.",
#     'Attendance': "Percentage of classes attended. Higher attendance usually helps.",
#     'Previous_Scores': "Scores from previous exams indicating past performance.",
#     'Sleep_Hours': "Average hours of sleep per night. Adequate sleep aids concentration.",
#     'Tutoring_Sessions': "Number of tutoring sessions attended monthly. Extra help improves learning.",
#     'Parental_Involvement': "Parental support in education: Low, Medium, High.",
#     'Access_to_Resources': "Availability of educational resources: Low, Medium, High.",
#     'Motivation_Level': "Student's motivation: Low, Medium, High.",
#     'Teacher_Quality': "Quality of teachers: Low, Medium, High.",
#     'School_Type': "Type of school: Public or Private.",
#     'Peer_Influence': "Peer impact on studies: Positive, Neutral, Negative.",
#     'Extracurricular_Activities': "Participation in extracurriculars: Yes or No.",
#     'Internet_Access': "Internet access availability: Yes or No.",
#     'Learning_Disabilities': "Presence of learning disabilities: Yes or No.",
#     'Parental_Education_Level': "Parents' education level: High School, College, Postgraduate.",
#     'Distance_from_Home': "Distance to school: Near, Moderate, Far.",
#     'Gender': "Student gender: Male or Female."
# }
#
# # Input widgets container with two columns for input and description
# for feature in numeric_features + categorical_features:
#     col1, col2 = st.columns([2, 3])
#     with col1:
#         if feature in numeric_features:
#             val = st.number_input(
#                 f"{feature.replace('_', ' ')}",
#                 value=float(df[feature].median()),
#                 step=0.1,
#                 key=feature
#             )
#         else:
#             options = sorted(df[feature].dropna().unique())
#             val = st.selectbox(
#                 f"{feature.replace('_', ' ')}",
#                 options,
#                 index=0,
#                 key=feature
#             )
#     with col2:
#         st.markdown(f"**Interpretation:** {feature_descriptions.get(feature, '')}")
#
#     # Store user input
#     if 'input_dict' not in st.session_state:
#         st.session_state['input_dict'] = {}
#     st.session_state['input_dict'][feature] = val
#
# # Build input DataFrame for prediction
# input_df = pd.DataFrame({k: [v] for k, v in st.session_state['input_dict'].items()})
#
# # Predict button
# if st.button("Predict Exam Score"):
#     pred_score = model.predict(input_df)[0]
#     st.success(f"🎉 Predicted Exam Score: **{pred_score:.2f}**")

#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import plotly.express as px
#
# st.set_page_config(page_title="Student Score Predictor Comparison", layout="wide")
# st.title("📊 Compare Student Exam Score Prediction Models")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df = pd.read_csv(csv_url)
#
# # Features for multi-feature model
# numeric_features = [
#     'Hours_Studied', 'Attendance', 'Previous_Scores',
#     'Sleep_Hours', 'Tutoring_Sessions'
# ]
# categorical_features = [
#     'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level',
#     'Teacher_Quality', 'School_Type', 'Peer_Influence',
#     'Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities',
#     'Parental_Education_Level', 'Distance_from_Home', 'Gender'
# ]
#
# # Clean data
# df = df.dropna(subset=['Exam_Score', 'Hours_Studied'] + numeric_features + categorical_features)
#
# # Prepare data for multi-feature model
# X_multi = df[numeric_features + categorical_features]
# y = df['Exam_Score']
#
# # Prepare data for simple model (Hours_Studied only)
# X_simple = df[['Hours_Studied']]
#
# # Split dataset for both models (same split for fair comparison)
# X_train_multi, X_test_multi, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)
# X_train_simple, X_test_simple, _, _ = train_test_split(X_simple, y, test_size=0.2, random_state=42)
#
# # Preprocessor for multi-feature model
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', 'passthrough', numeric_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ])
#
# # Pipelines
# multi_model = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', LinearRegression())
# ])
#
# simple_model = LinearRegression()
#
# # Train both models
# multi_model.fit(X_train_multi, y_train)
# simple_model.fit(X_train_simple, y_train)
#
# # Predictions on test sets
# y_pred_multi = multi_model.predict(X_test_multi)
# y_pred_simple = simple_model.predict(X_test_simple)
#
# # Metrics helper function
# def get_metrics(y_true, y_pred):
#     return {
#         "MAE": mean_absolute_error(y_true, y_pred),
#         "MSE": mean_squared_error(y_true, y_pred),
#         "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
#         "R2": r2_score(y_true, y_pred)
#     }
#
# metrics_multi = get_metrics(y_test, y_pred_multi)
# metrics_simple = get_metrics(y_test, y_pred_simple)
#
# # Display metrics side-by-side
# st.subheader("📈 Model Performance Metrics Comparison")
#
# col1, col2 = st.columns(2)
# with col1:
#     st.markdown("### Multi-Feature Linear Regression")
#     st.write(f"**MAE:** {metrics_multi['MAE']:.2f}")
#     st.write(f"**MSE:** {metrics_multi['MSE']:.2f}")
#     st.write(f"**RMSE:** {metrics_multi['RMSE']:.2f}")
#     st.write(f"**R² Score:** {metrics_multi['R2']:.2f}")
#
# with col2:
#     st.markdown("### Simple Linear Regression (Hours Studied Only)")
#     st.write(f"**MAE:** {metrics_simple['MAE']:.2f}")
#     st.write(f"**MSE:** {metrics_simple['MSE']:.2f}")
#     st.write(f"**RMSE:** {metrics_simple['RMSE']:.2f}")
#     st.write(f"**R² Score:** {metrics_simple['R2']:.2f}")
#
# # Visualization: Predicted vs Actual for both models
# st.subheader("🎯 Predicted vs Actual Exam Scores")
#
# fig = go.Figure()
# # Multi-feature model points
# fig.add_trace(go.Scatter(
#     x=y_test, y=y_pred_multi,
#     mode='markers',
#     name='Multi-Feature Model',
#     marker=dict(color='blue')
# ))
# # Simple model points
# fig.add_trace(go.Scatter(
#     x=y_test, y=y_pred_simple,
#     mode='markers',
#     name='Simple Model',
#     marker=dict(color='red')
# ))
# # Perfect prediction line
# min_score = min(y_test.min(), y_pred_multi.min(), y_pred_simple.min())
# max_score = max(y_test.max(), y_pred_multi.max(), y_pred_simple.max())
# fig.add_trace(go.Scatter(
#     x=[min_score, max_score],
#     y=[min_score, max_score],
#     mode='lines',
#     name='Perfect Prediction',
#     line=dict(color='green', dash='dash')
# ))
# fig.update_layout(
#     xaxis_title="Actual Exam Score",
#     yaxis_title="Predicted Exam Score",
#     legend=dict(orientation="h", y=-0.2),
#     height=500
# )
# st.plotly_chart(fig, use_container_width=True)
#
# st.markdown("---")
# st.header("📝 Predict Exam Score from Input Features")
#
# # Input for simple model
# hours_studied_simple = st.number_input(
#     "Enter Hours Studied (Simple Model):",
#     min_value=0.0, max_value=100.0, value=5.0, step=0.1,
#     key='hours_simple'
# )
# pred_simple = simple_model.predict(np.array([[hours_studied_simple]]))[0]
#
# # Input for multi-feature model (with explanations)
# st.markdown("### Multi-Feature Model Inputs")
#
# # Feature descriptions for user help
# feature_descriptions = {
#     'Hours_Studied': "Hours spent studying per week",
#     'Attendance': "Percentage of classes attended",
#     'Previous_Scores': "Scores from previous exams",
#     'Sleep_Hours': "Average hours of sleep per night",
#     'Tutoring_Sessions': "Number of tutoring sessions attended monthly",
#     'Parental_Involvement': "Parental support level (Low, Medium, High)",
#     'Access_to_Resources': "Availability of educational resources (Low, Medium, High)",
#     'Motivation_Level': "Motivation level (Low, Medium, High)",
#     'Teacher_Quality': "Teacher quality (Low, Medium, High)",
#     'School_Type': "Type of school (Public, Private)",
#     'Peer_Influence': "Peer influence (Positive, Neutral, Negative)",
#     'Extracurricular_Activities': "Participation (Yes, No)",
#     'Internet_Access': "Internet access (Yes, No)",
#     'Learning_Disabilities': "Learning disabilities (Yes, No)",
#     'Parental_Education_Level': "Parents' education level (High School, College, Postgraduate)",
#     'Distance_from_Home': "Distance to school (Near, Moderate, Far)",
#     'Gender': "Gender (Male, Female)"
# }
#
# input_data = {}
#
# for feature in numeric_features:
#     val = st.number_input(
#         f"{feature} ({feature_descriptions.get(feature)})",
#         value=float(df[feature].median()),
#         step=0.1,
#         key=f"multi_{feature}"
#     )
#     input_data[feature] = val
#
# for feature in categorical_features:
#     options = sorted(df[feature].dropna().unique())
#     val = st.selectbox(
#         f"{feature} ({feature_descriptions.get(feature)})",
#         options,
#         index=0,
#         key=f"multi_{feature}"
#     )
#     input_data[feature] = val
#
# # Predict multi-feature model result
# input_df = pd.DataFrame([input_data])
#
# if st.button("Predict Exam Scores for Both Models"):
#     pred_multi = multi_model.predict(input_df)[0]
#     st.success(f"Simple Model Prediction (Hours Studied only): **{pred_simple:.2f}**")
#     st.success(f"Multi-Feature Model Prediction: **{pred_multi:.2f}**")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

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
    subtab_main, subtab2, subtab3 = st.tabs(["Model & Visualizations", "Whole Dataset", "Test Data"])

    # Load dataset
    csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
    df = pd.read_csv(csv_url)

    # Clean data
    df_clean = df.dropna()
    # Simple model features
    X_simple = df_clean[['Hours_Studied']]
    y = df_clean['Exam_Score']

    # Multi-feature selection (including Hours_Studied + selected numeric columns)
    feature_cols = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']
    X_multi = df_clean[feature_cols]

    # Split datasets
    X_train_simple, X_test_simple, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
    X_train_multi, X_test_multi, _, _ = train_test_split(X_multi, y, test_size=0.2, random_state=42)

    # Sub-tabs: Linear | Multi-Feature
    with subtab_main:
        model_tab1, model_tab2 = st.tabs(["Simple Linear Model", "Multi-Feature Linear Model"])

        # ========================== SIMPLE LINEAR MODEL ==========================
        with model_tab1:
            st.subheader("🔵 Simple Linear Regression Model (Hours Studied only)")

            # Train simple linear model
            simple_model = LinearRegression()
            simple_model.fit(X_train_simple, y_train)
            y_pred_simple = simple_model.predict(X_test_simple)

            # Plot: Regression Line with actual data
            x_line = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
            y_line = simple_model.predict(x_line)
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_clean['Hours_Studied'], y=df_clean['Exam_Score'],
                                      mode='markers', name='Actual Data', marker=dict(color='blue')))
            fig1.add_trace(go.Scatter(x=x_line.flatten(), y=y_line,
                                      mode='lines', name='Regression Line', line=dict(color='red')))
            fig1.update_layout(title='📘 Hours Studied vs Exam Score (Simple Linear Model)',
                               xaxis_title='Hours Studied', yaxis_title='Exam Score')
            st.plotly_chart(fig1, use_container_width=True)

            # Evaluation
            st.subheader("📈 Simple Linear Model Metrics")
            st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_simple):.2f}")
            st.write(f"**MSE:** {mean_squared_error(y_test, y_pred_simple):.2f}")
            st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred_simple)):.2f}")
            st.write(f"**R² Score:** {r2_score(y_test, y_pred_simple):.2f}")
            st.write(f"**Equation:** `Exam_Score = {simple_model.intercept_:.2f} + {simple_model.coef_[0]:.2f} * Hours_Studied`")

            # Input prediction
            st.subheader("📝 Predict Exam Score for Custom Hours Studied (Simple Linear Model)")
            hours_input_simple = st.number_input("Enter hours studied:", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key="simple_linear_input")
            linear_pred_simple = simple_model.predict(np.array([[hours_input_simple]]))[0]
            st.write(f"**Prediction:** {linear_pred_simple:.2f} exam score")

        # ========================== MULTI-FEATURE LINEAR MODEL ==========================
        with model_tab2:
            st.subheader("🟢 Multi-Feature Linear Regression Model")

            # Train multi-feature linear model
            multi_model = LinearRegression()
            multi_model.fit(X_train_multi, y_train)
            y_pred_multi = multi_model.predict(X_test_multi)

            # For visualization — plot predicted vs actual but by Hours Studied only
            fig2 = px.scatter(x=y_test, y=y_pred_multi,
                              labels={'x': 'Actual Exam Score', 'y': 'Predicted Exam Score'},
                              title="🎯 Actual vs Predicted Exam Scores (Multi-Feature Model)")
            fig2.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                     y=[y_test.min(), y_test.max()],
                                     mode='lines', name='Perfect Fit', line=dict(color='red')))
            st.plotly_chart(fig2, use_container_width=True)

            # Evaluation
            st.subheader("📈 Multi-Feature Linear Model Metrics")
            st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_multi):.2f}")
            st.write(f"**MSE:** {mean_squared_error(y_test, y_pred_multi):.2f}")
            st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred_multi)):.2f}")
            st.write(f"**R² Score:** {r2_score(y_test, y_pred_multi):.2f}")

            st.write("**Coefficients:**")
            coef_df = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': multi_model.coef_
            })
            st.dataframe(coef_df)

            # Input prediction for multi-feature
            st.subheader("📝 Predict Exam Score for Custom Inputs (Multi-Feature Model)")
            hours_input_multi = st.number_input("Hours studied:", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key="multi_hours")
            attendance_input = st.slider("Attendance %:", 0, 100, 80, step=1)
            prev_score_input = st.slider("Previous Exam Score:", 0, 100, 75, step=1)
            sleep_input = st.slider("Sleep Hours per night:", 0.0, 12.0, 7.0, step=0.1)
            tutoring_input = st.slider("Monthly Tutoring Sessions:", 0, 20, 3, step=1)
            physical_input = st.slider("Physical Activity Hours per week:", 0.0, 20.0, 2.0, step=0.1)

            input_multi_array = np.array([[hours_input_multi, attendance_input, prev_score_input, sleep_input, tutoring_input, physical_input]])
            multi_pred = multi_model.predict(input_multi_array)[0]
            st.write(f"**Prediction:** {multi_pred:.2f} exam score")

        # ========================== MULTI-MODEL COMPARISON VISUALIZATION ==========================
        st.markdown("---")
        st.subheader("📊 Actual Data and Model Predictions Comparison")

        fig_all = go.Figure()

        # Actual data points
        fig_all.add_trace(go.Scatter(
            x=df_clean['Hours_Studied'], y=df_clean['Exam_Score'],
            mode='markers',
            name='Actual Data',
            marker=dict(color='white', size=6, symbol='circle')
        ))

        # Multi-feature model predictions (on test set)
        fig_all.add_trace(go.Scatter(
            x=X_test_multi['Hours_Studied'], y=y_pred_multi,
            mode='markers',
            name='Multi-Feature Predictions',
            marker=dict(color='blue', size=8, symbol='triangle-up')
        ))

        # Simple model predictions (on test set)
        fig_all.add_trace(go.Scatter(
            x=X_test_simple['Hours_Studied'], y=y_pred_simple,
            mode='markers',
            name='Simple Model Predictions',
            marker=dict(color='red', size=8, symbol='x')
        ))

        fig_all.update_layout(
            xaxis_title="Hours Studied",
            yaxis_title="Exam Score",
            legend=dict(orientation="h", y=-0.2),
            title="Comparison: Actual Data vs Predictions by Hours Studied",
            height=600
        )

        st.plotly_chart(fig_all, use_container_width=True)

    # ========================== WHOLE DATASET TAB ==========================
    with subtab2:
        st.subheader("📂 Full Dataset (Raw CSV)")
        if df.empty:
            st.warning("Dataset is empty or not loaded correctly.")
        else:
            st.dataframe(df, use_container_width=True)

    # ========================== TEST DATA TAB ==========================
    with subtab3:
        st.subheader("🧪 Test Data with Predictions")
        test_data_df = X_test_simple.copy()
        test_data_df['Actual'] = y_test.values
        test_data_df['Predicted (Simple Linear)'] = y_pred_simple
        test_data_df['Predicted (Multi-Feature)'] = y_pred_multi
        st.dataframe(test_data_df.reset_index(drop=True), use_container_width=True)
