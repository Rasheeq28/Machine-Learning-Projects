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


# linear
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
#     df_clean = df.dropna()
#     # Simple model features
#     X_simple = df_clean[['Hours_Studied']]
#     y = df_clean['Exam_Score']
#
#     # Multi-feature selection (including Hours_Studied + selected numeric columns)
#     feature_cols = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']
#     X_multi = df_clean[feature_cols]
#
#     # Split datasets
#     X_train_simple, X_test_simple, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
#     X_train_multi, X_test_multi, _, _ = train_test_split(X_multi, y, test_size=0.2, random_state=42)
#
#     # Sub-tabs: Linear | Multi-Feature
#     with subtab_main:
#         model_tab1, model_tab2 = st.tabs(["Simple Linear Model", "Multi-Feature Linear Model"])
#
#         # ========================== SIMPLE LINEAR MODEL ==========================
#         with model_tab1:
#             st.subheader("🔵 Simple Linear Regression Model (Hours Studied only)")
#
#             # Train simple linear model
#             simple_model = LinearRegression()
#             simple_model.fit(X_train_simple, y_train)
#             y_pred_simple = simple_model.predict(X_test_simple)
#
#             # Plot: Regression Line with actual data
#             x_line = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
#             y_line = simple_model.predict(x_line)
#             fig1 = go.Figure()
#             fig1.add_trace(go.Scatter(x=df_clean['Hours_Studied'], y=df_clean['Exam_Score'],
#                                       mode='markers', name='Actual Data', marker=dict(color='blue')))
#             fig1.add_trace(go.Scatter(x=x_line.flatten(), y=y_line,
#                                       mode='lines', name='Regression Line', line=dict(color='red')))
#             fig1.update_layout(title='📘 Hours Studied vs Exam Score (Simple Linear Model)',
#                                xaxis_title='Hours Studied', yaxis_title='Exam Score')
#             st.plotly_chart(fig1, use_container_width=True)
#
#             # Evaluation
#             st.subheader("📈 Simple Linear Model Metrics")
#             st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_simple):.2f}")
#             st.write(f"**MSE:** {mean_squared_error(y_test, y_pred_simple):.2f}")
#             st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred_simple)):.2f}")
#             st.write(f"**R² Score:** {r2_score(y_test, y_pred_simple):.2f}")
#             st.write(f"**Equation:** `Exam_Score = {simple_model.intercept_:.2f} + {simple_model.coef_[0]:.2f} * Hours_Studied`")
#
#             # Input prediction
#             st.subheader("📝 Predict Exam Score for Custom Hours Studied (Simple Linear Model)")
#             hours_input_simple = st.number_input("Enter hours studied:", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key="simple_linear_input")
#             linear_pred_simple = simple_model.predict(np.array([[hours_input_simple]]))[0]
#             st.write(f"**Prediction:** {linear_pred_simple:.2f} exam score")
#
#         # ========================== MULTI-FEATURE LINEAR MODEL ==========================
#         with model_tab2:
#             st.subheader("🟢 Multi-Feature Linear Regression Model")
#
#             # Train multi-feature linear model
#             multi_model = LinearRegression()
#             multi_model.fit(X_train_multi, y_train)
#             y_pred_multi = multi_model.predict(X_test_multi)
#
#             # For visualization — plot predicted vs actual but by Hours Studied only
#             fig2 = px.scatter(x=y_test, y=y_pred_multi,
#                               labels={'x': 'Actual Exam Score', 'y': 'Predicted Exam Score'},
#                               title="🎯 Actual vs Predicted Exam Scores (Multi-Feature Model)")
#             fig2.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
#                                      y=[y_test.min(), y_test.max()],
#                                      mode='lines', name='Perfect Fit', line=dict(color='red')))
#             st.plotly_chart(fig2, use_container_width=True)
#
#             # Evaluation
#             st.subheader("📈 Multi-Feature Linear Model Metrics")
#             st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_multi):.2f}")
#             st.write(f"**MSE:** {mean_squared_error(y_test, y_pred_multi):.2f}")
#             st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred_multi)):.2f}")
#             st.write(f"**R² Score:** {r2_score(y_test, y_pred_multi):.2f}")
#
#             st.write("**Coefficients:**")
#             coef_df = pd.DataFrame({
#                 'Feature': feature_cols,
#                 'Coefficient': multi_model.coef_
#             })
#             st.dataframe(coef_df)
#
#             # Input prediction for multi-feature
#             st.subheader("📝 Predict Exam Score for Custom Inputs (Multi-Feature Model)")
#             hours_input_multi = st.number_input("Hours studied:", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key="multi_hours")
#             attendance_input = st.slider("Attendance %:", 0, 100, 80, step=1)
#             prev_score_input = st.slider("Previous Exam Score:", 0, 100, 75, step=1)
#             sleep_input = st.slider("Sleep Hours per night:", 0.0, 12.0, 7.0, step=0.1)
#             tutoring_input = st.slider("Monthly Tutoring Sessions:", 0, 20, 3, step=1)
#             physical_input = st.slider("Physical Activity Hours per week:", 0.0, 20.0, 2.0, step=0.1)
#
#             input_multi_array = np.array([[hours_input_multi, attendance_input, prev_score_input, sleep_input, tutoring_input, physical_input]])
#             multi_pred = multi_model.predict(input_multi_array)[0]
#             st.write(f"**Prediction:** {multi_pred:.2f} exam score")
#
#         # ========================== MULTI-MODEL COMPARISON VISUALIZATION ==========================
#         st.markdown("---")
#         st.subheader("📊 Actual Data and Model Predictions Comparison")
#
#         fig_all = go.Figure()
#
#         # Actual data points
#         fig_all.add_trace(go.Scatter(
#             x=df_clean['Hours_Studied'], y=df_clean['Exam_Score'],
#             mode='markers',
#             name='Actual Data',
#             marker=dict(color='white', size=6, symbol='circle')
#         ))
#
#         # Multi-feature model predictions (on test set)
#         fig_all.add_trace(go.Scatter(
#             x=X_test_multi['Hours_Studied'], y=y_pred_multi,
#             mode='markers',
#             name='Multi-Feature Predictions',
#             marker=dict(color='blue', size=8, symbol='triangle-up')
#         ))
#
#         # Simple model predictions (on test set)
#         fig_all.add_trace(go.Scatter(
#             x=X_test_simple['Hours_Studied'], y=y_pred_simple,
#             mode='markers',
#             name='Simple Model Predictions',
#             marker=dict(color='red', size=8, symbol='x')
#         ))
#
#         fig_all.update_layout(
#             xaxis_title="Hours Studied",
#             yaxis_title="Exam Score",
#             legend=dict(orientation="h", y=-0.2),
#             title="Comparison: Actual Data vs Predictions by Hours Studied",
#             height=600
#         )
#
#         st.plotly_chart(fig_all, use_container_width=True)
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
#         st.subheader("🧪 Test Data with Predictions")
#         test_data_df = X_test_simple.copy()
#         test_data_df['Actual'] = y_test.values
#         test_data_df['Predicted (Simple Linear)'] = y_pred_simple
#         test_data_df['Predicted (Multi-Feature)'] = y_pred_multi
#         st.dataframe(test_data_df.reset_index(drop=True), use_container_width=True)


# all
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.cluster import DBSCAN, KMeans
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV
#
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
#     df_clean = df.dropna()
#     # Simple model features
#     X_simple = df_clean[['Hours_Studied']]
#     y = df_clean['Exam_Score']
#
#     # Multi-feature selection (including Hours_Studied + selected numeric columns)
#     feature_cols = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']
#     X_multi = df_clean[feature_cols]
#
#     # Split datasets
#     X_train_simple, X_test_simple, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
#     X_train_multi, X_test_multi, _, _ = train_test_split(X_multi, y, test_size=0.2, random_state=42)
#
#     # Sub-tabs: Linear | Multi-Feature | Simple Poly | Multi Poly
#     with subtab_main:
#         model_tab1, model_tab2, model_tab3, model_tab4 = st.tabs([
#             "Simple Linear Model",
#             "Multi-Feature Linear Model",
#             "Simple Polynomial Model",
#             "Multi-Feature Polynomial Model"
#         ])
#
#         # ========================== SIMPLE LINEAR MODEL ==========================
#         with model_tab1:
#             st.subheader("🔵 Simple Linear Regression Model (Hours Studied only)")
#
#             simple_model = LinearRegression()
#             simple_model.fit(X_train_simple, y_train)
#             y_pred_simple = simple_model.predict(X_test_simple)
#
#             # Plot regression line + actual data
#             x_line = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
#             y_line = simple_model.predict(x_line)
#             fig1 = go.Figure()
#             fig1.add_trace(go.Scatter(x=df_clean['Hours_Studied'], y=df_clean['Exam_Score'],
#                                       mode='markers', name='Actual Data', marker=dict(color='blue')))
#             fig1.add_trace(go.Scatter(x=x_line.flatten(), y=y_line,
#                                       mode='lines', name='Regression Line', line=dict(color='red')))
#             fig1.update_layout(title='📘 Hours Studied vs Exam Score (Simple Linear Model)',
#                                xaxis_title='Hours Studied', yaxis_title='Exam Score')
#             st.plotly_chart(fig1, use_container_width=True)
#
#             st.subheader("📈 Simple Linear Model Metrics")
#             st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_simple):.2f}")
#             st.write(f"**MSE:** {mean_squared_error(y_test, y_pred_simple):.2f}")
#             st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred_simple)):.2f}")
#             st.write(f"**R² Score:** {r2_score(y_test, y_pred_simple):.2f}")
#             st.write(f"**Equation:** `Exam_Score = {simple_model.intercept_:.2f} + {simple_model.coef_[0]:.2f} * Hours_Studied`")
#
#             st.subheader("📝 Predict Exam Score for Custom Hours Studied (Simple Linear Model)")
#             hours_input_simple = st.number_input("Enter hours studied:", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key="simple_linear_input")
#             linear_pred_simple = simple_model.predict(np.array([[hours_input_simple]]))[0]
#             st.write(f"**Prediction:** {linear_pred_simple:.2f} exam score")
#
#         # ========================== MULTI-FEATURE LINEAR MODEL ==========================
#         with model_tab2:
#             st.subheader("🟢 Multi-Feature Linear Regression Model")
#
#             multi_model = LinearRegression()
#             multi_model.fit(X_train_multi, y_train)
#             y_pred_multi = multi_model.predict(X_test_multi)
#
#             fig2 = px.scatter(x=y_test, y=y_pred_multi,
#                               labels={'x': 'Actual Exam Score', 'y': 'Predicted Exam Score'},
#                               title="🎯 Actual vs Predicted Exam Scores (Multi-Feature Model)")
#             fig2.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
#                                      y=[y_test.min(), y_test.max()],
#                                      mode='lines', name='Perfect Fit', line=dict(color='red')))
#             st.plotly_chart(fig2, use_container_width=True)
#
#             st.subheader("📈 Multi-Feature Linear Model Metrics")
#             st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_multi):.2f}")
#             st.write(f"**MSE:** {mean_squared_error(y_test, y_pred_multi):.2f}")
#             st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred_multi)):.2f}")
#             st.write(f"**R² Score:** {r2_score(y_test, y_pred_multi):.2f}")
#
#             st.write("**Coefficients:**")
#             coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': multi_model.coef_})
#             st.dataframe(coef_df)
#
#             st.subheader("📝 Predict Exam Score for Custom Inputs (Multi-Feature Model)")
#             hours_input_multi = st.number_input("Hours studied:", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key="multi_hours")
#             attendance_input = st.slider("Attendance %:", 0, 100, 80, step=1)
#             prev_score_input = st.slider("Previous Exam Score:", 0, 100, 75, step=1)
#             sleep_input = st.slider("Sleep Hours per night:", 0.0, 12.0, 7.0, step=0.1)
#             tutoring_input = st.slider("Monthly Tutoring Sessions:", 0, 20, 3, step=1)
#             physical_input = st.slider("Physical Activity Hours per week:", 0.0, 20.0, 2.0, step=0.1)
#
#             input_multi_array = np.array([[hours_input_multi, attendance_input, prev_score_input, sleep_input, tutoring_input, physical_input]])
#             multi_pred = multi_model.predict(input_multi_array)[0]
#             st.write(f"**Prediction:** {multi_pred:.2f} exam score")
#
#         # ========================== SIMPLE POLYNOMIAL MODEL ==========================
#         with model_tab3:
#             st.subheader("🟣 Simple Polynomial Regression Model (Degree 2, Hours Studied only)")
#
#             poly_simple_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
#             poly_simple_model.fit(X_train_simple, y_train)
#             y_pred_poly_simple = poly_simple_model.predict(X_test_simple)
#
#             # Plot polynomial curve + actual data
#             x_line = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
#             y_line_poly = poly_simple_model.predict(x_line)
#             fig3 = go.Figure()
#             fig3.add_trace(go.Scatter(x=df_clean['Hours_Studied'], y=df_clean['Exam_Score'],
#                                       mode='markers', name='Actual Data', marker=dict(color='blue')))
#             fig3.add_trace(go.Scatter(x=x_line.flatten(), y=y_line_poly,
#                                       mode='lines', name='Polynomial Curve', line=dict(color='green')))
#             fig3.update_layout(title='📗 Hours Studied vs Exam Score (Simple Polynomial Model)',
#                                xaxis_title='Hours Studied', yaxis_title='Exam Score')
#             st.plotly_chart(fig3, use_container_width=True)
#
#             st.subheader("📈 Simple Polynomial Model Metrics")
#             st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_poly_simple):.2f}")
#             st.write(f"**MSE:** {mean_squared_error(y_test, y_pred_poly_simple):.2f}")
#             st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred_poly_simple)):.2f}")
#             st.write(f"**R² Score:** {r2_score(y_test, y_pred_poly_simple):.2f}")
#             st.write("**Equation:** Polynomial equation based on degree 2 (not explicitly shown)")
#
#             st.subheader("📝 Predict Exam Score for Custom Hours Studied (Simple Polynomial Model)")
#             hours_input_poly_simple = st.number_input("Enter hours studied:", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key="simple_poly_input")
#             poly_pred_simple = poly_simple_model.predict(np.array([[hours_input_poly_simple]]))[0]
#             st.write(f"**Prediction:** {poly_pred_simple:.2f} exam score")
#
#         # ========================== MULTI-FEATURE POLYNOMIAL MODEL ==========================
#         with model_tab4:
#             st.subheader("🟠 Multi-Feature Polynomial Regression Model (Degree 2)")
#
#             poly_multi_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
#             poly_multi_model.fit(X_train_multi, y_train)
#             y_pred_poly_multi = poly_multi_model.predict(X_test_multi)
#
#             fig4 = px.scatter(x=y_test, y=y_pred_poly_multi,
#                               labels={'x': 'Actual Exam Score', 'y': 'Predicted Exam Score'},
#                               title="🎯 Actual vs Predicted Exam Scores (Multi-Feature Polynomial Model)")
#             fig4.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
#                                      y=[y_test.min(), y_test.max()],
#                                      mode='lines', name='Perfect Fit', line=dict(color='red')))
#             st.plotly_chart(fig4, use_container_width=True)
#
#             st.subheader("📈 Multi-Feature Polynomial Model Metrics")
#             st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_poly_multi):.2f}")
#             st.write(f"**MSE:** {mean_squared_error(y_test, y_pred_poly_multi):.2f}")
#             st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred_poly_multi)):.2f}")
#             st.write(f"**R² Score:** {r2_score(y_test, y_pred_poly_multi):.2f}")
#             st.write("**Equation:** Polynomial equation based on degree 2 (not explicitly shown)")
#
#             st.subheader("📝 Predict Exam Score for Custom Inputs (Multi-Feature Polynomial Model)")
#             hours_input_poly_multi = st.number_input("Hours studied:", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key="poly_multi_hours")
#             attendance_input_poly = st.slider("Attendance %:", 0, 100, 80, step=1, key="poly_attendance")
#             prev_score_input_poly = st.slider("Previous Exam Score:", 0, 100, 75, step=1, key="poly_prev_score")
#             sleep_input_poly = st.slider("Sleep Hours per night:", 0.0, 12.0, 7.0, step=0.1, key="poly_sleep")
#             tutoring_input_poly = st.slider("Monthly Tutoring Sessions:", 0, 20, 3, step=1, key="poly_tutoring")
#             physical_input_poly = st.slider("Physical Activity Hours per week:", 0.0, 20.0, 2.0, step=0.1, key="poly_physical")
#
#             input_poly_multi_array = np.array([[hours_input_poly_multi, attendance_input_poly, prev_score_input_poly, sleep_input_poly, tutoring_input_poly, physical_input_poly]])
#             poly_multi_pred = poly_multi_model.predict(input_poly_multi_array)[0]
#             st.write(f"**Prediction:** {poly_multi_pred:.2f} exam score")
#
#         # ========================== MULTI-MODEL COMPARISON VISUALIZATION ==========================
#         st.markdown("---")
#         st.subheader("📊 Actual Data and Model Predictions Comparison")
#
#         fig_all = go.Figure()
#
#         # Actual data points
#         fig_all.add_trace(go.Scatter(
#             x=df_clean['Hours_Studied'], y=df_clean['Exam_Score'],
#             mode='markers',
#             name='Actual Data',
#             marker=dict(color='white', size=6, symbol='circle')
#         ))
#
#         # Multi-feature linear model predictions (on test set)
#         fig_all.add_trace(go.Scatter(
#             x=X_test_multi['Hours_Studied'], y=y_pred_multi,
#             mode='markers',
#             name='Multi-Feature Linear Predictions',
#             marker=dict(color='blue', size=8, symbol='triangle-up')
#         ))
#
#         # Simple linear model predictions (on test set)
#         fig_all.add_trace(go.Scatter(
#             x=X_test_simple['Hours_Studied'], y=y_pred_simple,
#             mode='markers',
#             name='Simple Linear Predictions',
#             marker=dict(color='red', size=8, symbol='x')
#         ))
#
#         # Simple polynomial model predictions (on test set)
#         fig_all.add_trace(go.Scatter(
#             x=X_test_simple['Hours_Studied'], y=y_pred_poly_simple,
#             mode='markers',
#             name='Simple Polynomial Predictions',
#             marker=dict(color='purple', size=8, symbol='diamond')
#         ))
#
#         # Multi-feature polynomial model predictions (on test set)
#         fig_all.add_trace(go.Scatter(
#             x=X_test_multi['Hours_Studied'], y=y_pred_poly_multi,
#             mode='markers',
#             name='Multi-Feature Polynomial Predictions',
#             marker=dict(color='orange', size=8, symbol='star')
#         ))
#
#         fig_all.update_layout(
#             xaxis_title="Hours Studied",
#             yaxis_title="Exam Score",
#             legend=dict(orientation="h", y=-0.2),
#             title="Comparison: Actual Data vs Predictions by Hours Studied",
#             height=600
#         )
#
#         st.plotly_chart(fig_all, use_container_width=True)
#
# # ========================== WHOLE DATASET TAB ==========================
# with subtab2:
#     st.subheader("📂 Full Dataset (Raw CSV)")
#     if df.empty:
#         st.warning("Dataset is empty or not loaded correctly.")
#     else:
#         st.dataframe(df, use_container_width=True)
#
# # ========================== TEST DATA TAB ==========================
# with tab2:
#     st.subheader("🧍 Customer Segmentation using Clustering")
#
#     # Create subtabs
#     data_tab, viz_tab = st.tabs(["📂 Dataset & Preprocessing", "📊 Visualizations & Interpretations"])
#
#     with data_tab:
#         # Load dataset from URL
#         df_customers = pd.read_csv("https://raw.githubusercontent.com/Rasheeq28/datasets/refs/heads/main/Mall_Customers.csv")
#
#         st.write("### Raw Dataset Preview")
#         st.dataframe(df_customers.head(), use_container_width=True)
#
#         # Select relevant features: Annual Income and Spending Score
#         df_selected = df_customers[['Annual Income (k$)', 'Spending Score (1-100)']]
#
#         # Standardize features
#         scaler = StandardScaler()
#         df_scaled = scaler.fit_transform(df_selected)
#
#         st.write("### Scaled Features Preview")
#         st.dataframe(pd.DataFrame(df_scaled, columns=['Annual Income (scaled)', 'Spending Score (scaled)']), use_container_width=True)
#
#     with viz_tab:
#         # Elbow method to find optimal number of clusters
#         inertia = []
#         k_range = range(1, 11)
#         for k in k_range:
#             kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#             kmeans.fit(df_scaled)
#             inertia.append(kmeans.inertia_)
#
#         fig_elbow = px.line(
#             x=list(k_range),
#             y=inertia,
#             labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'},
#             title="🔍 Elbow Method for Optimal K"
#         )
#         st.plotly_chart(fig_elbow, use_container_width=True)
#
#         st.write("""
#         **Interpretation:**
#         The elbow plot shows the inertia decreasing as K increases. The 'elbow' point suggests the optimal number of clusters.
#         Here, K=5 is chosen to balance complexity and fit.
#         """)
#
#         # KMeans clustering with optimal_k=5
#         optimal_k = 5
#         kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
#         df_customers['Cluster'] = kmeans.fit_predict(df_scaled)
#
#         fig_cluster = px.scatter(
#             df_customers,
#             x='Annual Income (k$)',
#             y='Spending Score (1-100)',
#             color=df_customers['Cluster'].astype(str),
#             title="💠 Customer Clusters (KMeans)",
#             labels={'Cluster': 'Segment'},
#             template="plotly"
#         )
#         st.plotly_chart(fig_cluster, use_container_width=True)
#
#         st.write("""
#         **Cluster Insights:**
#         - Colors represent distinct segments based on income and spending.
#         - Use to tailor marketing: e.g., premium offers for high-income/high-spenders.
#         """)
#
#         # DBSCAN clustering
#         dbscan = DBSCAN(eps=0.6, min_samples=5)
#         df_customers['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)
#
#         fig_dbscan = px.scatter(
#             df_customers,
#             x='Annual Income (k$)',
#             y='Spending Score (1-100)',
#             color=df_customers['DBSCAN_Cluster'].astype(str),
#             title="🔷 DBSCAN Clustering Results",
#             labels={'DBSCAN_Cluster': 'Segment'},
#             template="plotly_dark"
#         )
#         st.plotly_chart(fig_dbscan, use_container_width=True)
#
#         st.write("""
#         **DBSCAN Analysis:**
#         - Detects clusters by density and highlights outliers (-1).
#         - Outliers could be niche customers or anomalies.
#         """)
#
#         # Cluster averages summary (KMeans)
#         st.write("### 🧾 Average Spending per Cluster (KMeans)")
#         cluster_summary = df_customers.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
#         st.dataframe(cluster_summary)
#
#         st.write("""
#         **Cluster Averages Interpretation:**
#         - Shows average income & spending per cluster.
#         - Identify high-value segments for focused engagement.
#         """)
#
#         # Cluster sizes
#         st.write("### 📊 Cluster Sizes")
#         cluster_counts = df_customers['Cluster'].value_counts().sort_index()
#         st.bar_chart(cluster_counts)
#
#         st.write("""
#         **Cluster Size Insights:**
#         - Larger clusters mean bigger customer groups; smaller ones are niches.
#         - Allocate marketing resources accordingly.
#         """)
#
#         st.markdown("---")
#         st.write("🔄 **Note:** Features were standardized to give equal importance before clustering.")


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.express as px
# import plotly.graph_objects as go
#
# # ========================= STREAMLIT PAGE CONFIG =========================
# st.set_page_config(page_title="📊 Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # ========================= LOAD DATA =========================
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df = pd.read_csv(csv_url)
# df_clean = df.dropna()
#
# # ========================= FEATURES =========================
# target = "Exam_Score"
# single_feature = ["Hours_Studied"]
# all_features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
#     "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
#     "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
#     "Parental_Education_Level", "Distance_from_Home", "Gender"
# ]
#
# # Separate categorical and numeric columns
# numeric_features = df_clean[all_features].select_dtypes(include=["int64", "float64"]).columns.tolist()
# categorical_features = list(set(all_features) - set(numeric_features))
#
# # ========================= DATA SPLIT =========================
# X_single = df_clean[single_feature]
# X_multi = df_clean[all_features]
# y = df_clean[target]
#
# X_train_single, X_test_single, y_train, y_test = train_test_split(
#     X_single, y, test_size=0.2, random_state=42
# )
# X_train_multi, X_test_multi, _, _ = train_test_split(
#     X_multi, y, test_size=0.2, random_state=42
# )
#
# # Preprocessing for multi-feature (encode categoricals + scale numerics)
# preprocessor_multi = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), numeric_features),
#         ("cat", OneHotEncoder(drop="first"), categorical_features)
#     ]
# )
#
# # ========================= STREAMLIT TABS =========================
# tab1, tab2, tab3, tab4 = st.tabs([
#     "Simple Linear Model",
#     "Multi-Feature Linear Model",
#     "Simple Polynomial Model",
#     "Multi-Feature Polynomial Model"
# ])
#
# # ========== TAB 1: SIMPLE LINEAR ==========
# with tab1:
#     st.subheader("🔵 Simple Linear Regression (Hours Studied only)")
#
#     simple_model = LinearRegression()
#     simple_model.fit(X_train_single, y_train)
#     y_pred_single = simple_model.predict(X_test_single)
#
#     # Plot regression line
#     x_line = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
#     y_line = simple_model.predict(x_line)
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df_clean["Hours_Studied"], y=df_clean["Exam_Score"],
#         mode="markers", name="Actual", marker=dict(color="blue")
#     ))
#     fig.add_trace(go.Scatter(
#         x=x_line.flatten(), y=y_line,
#         mode="lines", name="Regression Line", line=dict(color="red")
#     ))
#     fig.update_layout(title="Hours Studied vs Exam Score", xaxis_title="Hours Studied", yaxis_title="Exam Score")
#     st.plotly_chart(fig, use_container_width=True)
#
#     # Metrics
#     st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_single):.2f}")
#     st.write(f"**R²:** {r2_score(y_test, y_pred_single):.2f}")
#     st.write(f"**Equation:** `Exam_Score = {simple_model.intercept_:.2f} + {simple_model.coef_[0]:.2f} * Hours_Studied`")
#
# # ========== TAB 2: MULTI-FEATURE LINEAR ==========
# with tab2:
#     st.subheader("🟢 Multi-Feature Linear Regression")
#
#     multi_model = Pipeline([
#         ("preprocessor", preprocessor_multi),
#         ("regressor", LinearRegression())
#     ])
#     multi_model.fit(X_train_multi, y_train)
#     y_pred_multi = multi_model.predict(X_test_multi)
#
#     fig = px.scatter(x=y_test, y=y_pred_multi, labels={"x": "Actual", "y": "Predicted"},
#                      title="Actual vs Predicted (Multi-Feature Linear)")
#     fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
#                              mode="lines", name="Perfect Fit", line=dict(color="red")))
#     st.plotly_chart(fig, use_container_width=True)
#
#     st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_multi):.2f}")
#     st.write(f"**R²:** {r2_score(y_test, y_pred_multi):.2f}")
#
# # ========== TAB 3: SIMPLE POLYNOMIAL ==========
# with tab3:
#     st.subheader("🟣 Simple Polynomial Regression (Hours Studied only, Degree 2)")
#
#     poly_simple_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
#     poly_simple_model.fit(X_train_single, y_train)
#     y_pred_poly_single = poly_simple_model.predict(X_test_single)
#
#     x_line_poly = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
#     y_line_poly = poly_simple_model.predict(x_line_poly)
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df_clean["Hours_Studied"], y=df_clean["Exam_Score"],
#         mode="markers", name="Actual", marker=dict(color="blue")
#     ))
#     fig.add_trace(go.Scatter(
#         x=x_line_poly.flatten(), y=y_line_poly,
#         mode="lines", name="Polynomial Curve", line=dict(color="green")
#     ))
#     fig.update_layout(title="Hours Studied vs Exam Score (Polynomial)", xaxis_title="Hours Studied", yaxis_title="Exam Score")
#     st.plotly_chart(fig, use_container_width=True)
#
#     st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_poly_single):.2f}")
#     st.write(f"**R²:** {r2_score(y_test, y_pred_poly_single):.2f}")
#
# # ========== TAB 4: MULTI-FEATURE POLYNOMIAL ==========
# with tab4:
#     st.subheader("🟠 Multi-Feature Polynomial Regression (Degree 2)")
#
#     poly_multi_model = Pipeline([
#         ("preprocessor", preprocessor_multi),
#         ("poly", PolynomialFeatures(degree=2, include_bias=False)),
#         ("regressor", LinearRegression())
#     ])
#     poly_multi_model.fit(X_train_multi, y_train)
#     y_pred_poly_multi = poly_multi_model.predict(X_test_multi)
#
#     fig = px.scatter(x=y_test, y=y_pred_poly_multi, labels={"x": "Actual", "y": "Predicted"},
#                      title="Actual vs Predicted (Multi-Feature Polynomial)")
#     fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
#                              mode="lines", name="Perfect Fit", line=dict(color="red")))
#     st.plotly_chart(fig, use_container_width=True)
#
#     st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred_poly_multi):.2f}")
#     st.write(f"**R²:** {r2_score(y_test, y_pred_poly_multi):.2f}")

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.express as px
# import plotly.graph_objects as go
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df = pd.read_csv(csv_url)
# df_clean = df.dropna()
#
# target = "Exam_Score"
# single_feature = ["Hours_Studied"]
# all_features = [col for col in df_clean.columns if col != target]
#
# # Split
# X_single = df_clean[single_feature]
# X_multi = df_clean[all_features]
# y = df_clean[target]
#
# X_train_s, X_test_s, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)
# X_train_m, X_test_m, _, _ = train_test_split(X_multi, y, test_size=0.2, random_state=42)
#
# # Preprocessing for multi-feature
# numeric_cols = X_multi.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = [c for c in all_features if c not in numeric_cols]
#
# preprocessor = ColumnTransformer([
#     ("num", StandardScaler(), numeric_cols),
#     ("cat", OneHotEncoder(drop="first"), cat_cols)
# ])
#
# # Models
# models = {
#     "Simple Linear": Pipeline([("lr", LinearRegression())]),
#     "Multi-Feature Linear": Pipeline([("prep", preprocessor), ("lr", LinearRegression())]),
#     "Simple Poly (deg=2)": Pipeline([("poly", PolynomialFeatures(degree=2)), ("lr", LinearRegression())]),
#     "Multi-Feature Poly (deg=2)": Pipeline([
#         ("prep", preprocessor),
#         ("poly", PolynomialFeatures(degree=2, include_bias=False)),
#         ("lr", LinearRegression())
#     ])
# }
#
# # Train & Predict
# predictions = {}
# metrics = []
# for name, model in models.items():
#     Xtr = X_train_s if "Simple" in name else X_train_m
#     Xte = X_test_s if "Simple" in name else X_test_m
#     model.fit(Xtr, y_train)
#     y_pred = model.predict(Xte)
#     predictions[name] = (Xte.copy(), y_pred)
#     metrics.append([name,
#                     r2_score(y_test, y_pred),
#                     mean_absolute_error(y_test, y_pred),
#                     mean_squared_error(y_test, y_pred),
#                     np.sqrt(mean_squared_error(y_test, y_pred))])
#
# metrics_df = pd.DataFrame(metrics, columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.subheader("Model Performance Comparison")
# st.dataframe(metrics_df.set_index("Model"))
#
# # Plot comparison (Actual vs Predictions)
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))
# for name, (Xte, y_pred) in predictions.items():
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df = pd.read_csv(csv_url)
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         # fallback if mode is empty (rare case)
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Features to use (including Hours_Studied and others you specified)
# features = [
#     "Hours_Studied",
#     "Attendance",
#     "Parental_Involvement",
#     "Access_to_Resources",
#     "Extracurricular_Activities",
#     "Sleep_Hours",
#     "Previous_Scores",
#     "Motivation_Level",
#     "Internet_Access",
#     "Tutoring_Sessions",
#     "Family_Income",
#     "Teacher_Quality",
#     "School_Type",
#     "Peer_Influence",
#     "Physical_Activity",
#     "Learning_Disabilities",
#     "Parental_Education_Level",
#     "Distance_from_Home",
#     "Gender"
# ]
#
# # Filter features that actually exist in the dataset (avoid typos or missing columns)
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # For comparison, keep single feature dataset (Hours_Studied only)
# X_single = df[["Hours_Studied"]]
#
# # Split
# X_train_s, X_test_s, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)
# X_train_m, X_test_m, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Preprocessing for multi-feature: separate numeric and categorical
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = [c for c in features if c not in numeric_cols]
#
# preprocessor = ColumnTransformer([
#     ("num", StandardScaler(), numeric_cols),
#     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
# ])
#
# # Define models
# models = {
#     "Simple Linear": Pipeline([("lr", LinearRegression())]),
#     "Multi-Feature Linear": Pipeline([("prep", preprocessor), ("lr", LinearRegression())]),
#     "Simple Poly (deg=2)": Pipeline([("poly", PolynomialFeatures(degree=2)), ("lr", LinearRegression())]),
#     "Multi-Feature Poly (deg=2)": Pipeline([
#         ("prep", preprocessor),
#         ("poly", PolynomialFeatures(degree=2, include_bias=False)),
#         ("lr", LinearRegression())
#     ])
# }
#
# # Train & Predict
# predictions = {}
# metrics = []
# for name, model in models.items():
#     Xtr = X_train_s if "Simple" in name else X_train_m
#     Xte = X_test_s if "Simple" in name else X_test_m
#     model.fit(Xtr, y_train)
#     y_pred = model.predict(Xte)
#     predictions[name] = (Xte.copy(), y_pred)
#     metrics.append([
#         name,
#         r2_score(y_test, y_pred),
#         mean_absolute_error(y_test, y_pred),
#         mean_squared_error(y_test, y_pred),
#         np.sqrt(mean_squared_error(y_test, y_pred))
#     ])
#
# metrics_df = pd.DataFrame(metrics, columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.subheader("Model Performance Comparison")
# st.dataframe(metrics_df.set_index("Model"))
#
# # Plot comparison (Actual vs Predictions)
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))
# for name, (Xte, y_pred) in predictions.items():
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)

#
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df = pd.read_csv(csv_url)
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Define features explicitly from your sample
# features = [
#     "Hours_Studied",
#     "Attendance",
#     "Parental_Involvement",
#     "Access_to_Resources",
#     "Extracurricular_Activities",
#     "Sleep_Hours",
#     "Previous_Scores",
#     "Motivation_Level",
#     "Internet_Access",
#     "Tutoring_Sessions",
#     "Family_Income",
#     "Teacher_Quality",
#     "School_Type",
#     "Peer_Influence",
#     "Physical_Activity",
#     "Learning_Disabilities",
#     "Parental_Education_Level",
#     "Distance_from_Home",  # categorical like Near, Moderate
#     "Gender"
# ]
#
# # Filter features actually in dataframe
# features = [f for f in features if f in df.columns]
#
# # ----------- FEATURE ENGINEERING -----------
#
# # 1. Encode Gender to numeric (Male=0, Female=1)
# if 'Gender' in df.columns and df['Gender'].dtype == 'object':
#     df['Gender_Encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
#     features = [f if f != 'Gender' else 'Gender_Encoded' for f in features]
#
# # 2. Interaction features for numeric columns: multiply Hours_Studied by numeric features
# # Identify numeric features (likely: Hours_Studied, Sleep_Hours, Previous_Scores, Tutoring_Sessions)
# numeric_features = ['Hours_Studied', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions']
# for feat in numeric_features:
#     if feat in df.columns and feat != 'Hours_Studied':
#         df[f'Hours_Studied_x_{feat}'] = df['Hours_Studied'] * df[feat]
#
# # 3. Interaction categorical feature between Family_Income and Access_to_Resources
# if 'Family_Income' in df.columns and 'Access_to_Resources' in df.columns:
#     df['FamilyIncome_x_Resources'] = df['Family_Income'].astype(str) + "_" + df['Access_to_Resources'].astype(str)
#
# # 4. Interaction categorical feature between Attendance and Motivation_Level
# if 'Attendance' in df.columns and 'Motivation_Level' in df.columns:
#     df['Attendance_x_Motivation'] = df['Attendance'].astype(str) + "_" + df['Motivation_Level'].astype(str)
#
# # Add these new engineered features to the feature list
# new_features = [f'Hours_Studied_x_{feat}' for feat in numeric_features if feat != 'Hours_Studied']
# if 'FamilyIncome_x_Resources' in df.columns:
#     new_features.append('FamilyIncome_x_Resources')
# if 'Attendance_x_Motivation' in df.columns:
#     new_features.append('Attendance_x_Motivation')
#
# all_features = features + new_features
#
# # ----------- END FEATURE ENGINEERING -----------
#
# X = df[all_features]
# y = df[target]
#
# # Single feature dataset for baseline
# X_single = df[["Hours_Studied"]]
#
# # Train-test split
# X_train_s, X_test_s, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)
# X_train_m, X_test_m, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Separate numeric and categorical for preprocessing
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = [c for c in all_features if c not in numeric_cols]
#
# preprocessor = ColumnTransformer([
#     ("num", StandardScaler(), numeric_cols),
#     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
# ])
#
# # Models
# models = {
#     "Simple Linear": Pipeline([("lr", LinearRegression())]),
#     "Multi-Feature Linear": Pipeline([("prep", preprocessor), ("lr", LinearRegression())]),
#     "Simple Poly (deg=2)": Pipeline([("poly", PolynomialFeatures(degree=2)), ("lr", LinearRegression())]),
#     "Multi-Feature Poly (deg=2)": Pipeline([
#         ("prep", preprocessor),
#         ("poly", PolynomialFeatures(degree=2, include_bias=False)),
#         ("lr", LinearRegression())
#     ])
# }
#
# # Train, predict, and collect metrics
# predictions = {}
# metrics = []
# for name, model in models.items():
#     Xtr = X_train_s if "Simple" in name else X_train_m
#     Xte = X_test_s if "Simple" in name else X_test_m
#     model.fit(Xtr, y_train)
#     y_pred = model.predict(Xte)
#     predictions[name] = (Xte.copy(), y_pred)
#     metrics.append([
#         name,
#         r2_score(y_test, y_pred),
#         mean_absolute_error(y_test, y_pred),
#         mean_squared_error(y_test, y_pred),
#         np.sqrt(mean_squared_error(y_test, y_pred))
#     ])
#
# metrics_df = pd.DataFrame(metrics, columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.subheader("Model Performance Comparison")
# st.dataframe(metrics_df.set_index("Model"))
#
# # Plot Actual vs Predicted
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))
# for name, (Xte, y_pred) in predictions.items():
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)

# better polynomial
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import warnings
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df = pd.read_csv(csv_url)
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
#
# # Filter features actually in dataframe
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # Separate numeric and categorical columns for preprocessing
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# # Define preprocessing pipelines
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('scaler', StandardScaler())
# ])
#
# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('num', numeric_poly_transformer, numeric_cols),
#         ('cat', categorical_transformer, cat_cols)
#     ],
#     remainder='passthrough'
# )
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', StandardScaler(), numeric_cols),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Define models
# models = {
#     "Simple Linear Regression": Pipeline(steps=[
#         ('preprocessor', ColumnTransformer(
#             transformers=[
#                 ('scaler', StandardScaler(), ['Hours_Studied'])
#             ],
#             remainder='drop'
#         )),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Linear Regression": Pipeline(steps=[
#         ('preprocessor', preprocessor_linear),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Polynomial (deg=2)": Pipeline(steps=[
#         ('preprocessor', preprocessor_poly),
#         ('regressor', LinearRegression())
#     ])
# }
#
# # Cross-validation results
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# for name, model in models.items():
#     r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
#     mae_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
#     mse_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
#     rmse_scores = np.sqrt(mse_scores)
#
#     cv_results.append([
#         name,
#         np.mean(r2_scores),
#         np.std(r2_scores),
#         np.mean(mae_scores),
#         np.mean(mse_scores),
#         np.mean(rmse_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV R²", "Std Dev CV R²", "Mean CV MAE", "Mean CV MSE", "Mean CV RMSE"])
# st.dataframe(cv_df.set_index("Model"))
#
# # Train final models and plot Actual vs Predicted
# st.subheader("Actual vs Predicted Scores on Test Set")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))
#
# predictions = {}
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     predictions[name] = y_pred
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
#
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, model in models.items():
#     y_pred = predictions[name]
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#
#     test_metrics.append([name, r2, mae, mse, rmse])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.dataframe(test_metrics_df.set_index("Model"))

# better
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import warnings
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df = pd.read_csv(csv_url)
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
#
# # Filter features actually in dataframe
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # Separate numeric and categorical columns for preprocessing
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# # Define preprocessing pipelines for multi-feature models
# # The numeric pipeline applies PolynomialFeatures and then StandardScaler
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('scaler', StandardScaler())
# ])
#
# # The categorical pipeline applies OneHotEncoder
# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# # ColumnTransformer for polynomial model
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('num', numeric_poly_transformer, numeric_cols),
#         ('cat', categorical_transformer, cat_cols)
#     ],
#     remainder='passthrough'
# )
#
# # ColumnTransformer for linear model
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', StandardScaler(), numeric_cols),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Define models with robust pipelines
# models = {
#     "Simple Linear Regression": Pipeline(steps=[
#         ('scaler', StandardScaler()),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Linear Regression": Pipeline(steps=[
#         ('preprocessor', preprocessor_linear),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Polynomial (deg=2)": Pipeline(steps=[
#         ('preprocessor', preprocessor_poly),
#         ('regressor', LinearRegression())
#     ])
# }
#
# # Cross-validation results
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# for name, model in models.items():
#     # For the simple model, we need to pass only the "Hours_Studied" column
#     if name == "Simple Linear Regression":
#         X_cv = X[["Hours_Studied"]]
#     else:
#         X_cv = X
#
#     r2_scores = cross_val_score(model, X_cv, y, cv=5, scoring='r2')
#     mae_scores = -cross_val_score(model, X_cv, y, cv=5, scoring='neg_mean_absolute_error')
#     mse_scores = -cross_val_score(model, X_cv, y, cv=5, scoring='neg_mean_squared_error')
#     rmse_scores = np.sqrt(mse_scores)
#
#     cv_results.append([
#         name,
#         np.mean(r2_scores),
#         np.std(r2_scores),
#         np.mean(mae_scores),
#         np.mean(mse_scores),
#         np.mean(rmse_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV R²", "Std Dev CV R²", "Mean CV MAE", "Mean CV MSE", "Mean CV RMSE"])
# st.dataframe(cv_df.set_index("Model"))
#
# # Train final models and plot Actual vs Predicted
# st.subheader("Actual vs Predicted Scores on Test Set")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))
#
# predictions = {}
# for name, model in models.items():
#     # For the simple model, train on the single feature
#     if name == "Simple Linear Regression":
#         X_train_specific = X_train[["Hours_Studied"]]
#         X_test_specific = X_test[["Hours_Studied"]]
#     else:
#         X_train_specific = X_train
#         X_test_specific = X_test
#
#     model.fit(X_train_specific, y_train)
#     y_pred = model.predict(X_test_specific)
#     predictions[name] = y_pred
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
#
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, y_pred in predictions.items():
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#
#     test_metrics.append([name, r2, mae, mse, rmse])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.dataframe(test_metrics_df.set_index("Model"))
#
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import warnings
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df_raw = pd.read_csv(csv_url)
# df = df_raw.copy()
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # --- REFINED PREPROCESSING PIPELINES ---
# # Instead of manual encoding, we'll let OneHotEncoder handle categorical features.
# # This avoids making assumptions about the ordinal nature of data.
#
# # Separate numeric and categorical columns for preprocessing
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# # Define preprocessing pipelines for multi-feature models
# # The numeric pipeline applies PolynomialFeatures and then StandardScaler
# # We will use a smaller subset of numeric features for the polynomial model
# # to avoid overfitting and multicollinearity.
# poly_features_list = ["Hours_Studied", "Previous_Scores", "Sleep_Hours"]
#
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('scaler', StandardScaler())
# ])
#
# # For all other numeric columns, just scale them
# numeric_scaler = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])
#
# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# # Combine transformers into ColumnTransformers
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('poly_num', numeric_poly_transformer, poly_features_list),
#         ('other_num', numeric_scaler, [col for col in numeric_cols if col not in poly_features_list]),
#         ('cat', categorical_transformer, cat_cols)
#     ],
#     remainder='passthrough'
# )
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', StandardScaler(), numeric_cols),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Define models with robust pipelines
# models = {
#     "Simple Linear Regression": Pipeline(steps=[
#         ('scaler', StandardScaler()),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Linear Regression": Pipeline(steps=[
#         ('preprocessor', preprocessor_linear),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Polynomial (deg=2)": Pipeline(steps=[
#         ('preprocessor', preprocessor_poly),
#         ('regressor', LinearRegression())
#     ])
# }
#
# # Cross-validation results
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         X_cv = X[["Hours_Studied"]]
#     else:
#         X_cv = X
#
#     r2_scores = cross_val_score(model, X_cv, y, cv=5, scoring='r2')
#     mae_scores = -cross_val_score(model, X_cv, y, cv=5, scoring='neg_mean_absolute_error')
#     mse_scores = -cross_val_score(model, X_cv, y, cv=5, scoring='neg_mean_squared_error')
#     rmse_scores = np.sqrt(mse_scores)
#
#     cv_results.append([
#         name,
#         np.mean(r2_scores),
#         np.std(r2_scores),
#         np.mean(mae_scores),
#         np.mean(mse_scores),
#         np.mean(rmse_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV R²", "Std Dev CV R²", "Mean CV MAE", "Mean CV MSE", "Mean CV RMSE"])
# st.dataframe(cv_df.set_index("Model"))
#
# # Train final models and plot Actual vs Predicted
# st.subheader("Actual vs Predicted Scores on Test Set")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))
#
# predictions = {}
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         X_train_specific = X_train[["Hours_Studied"]]
#         X_test_specific = X_test[["Hours_Studied"]]
#     else:
#         X_train_specific = X_train
#         X_test_specific = X_test
#
#     model.fit(X_train_specific, y_train)
#     y_pred = model.predict(X_test_specific)
#     predictions[name] = y_pred
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
#
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, y_pred in predictions.items():
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#
#     test_metrics.append([name, r2, mae, mse, rmse])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.dataframe(test_metrics_df.set_index("Model"))



# RIDGE
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import warnings
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df_raw = pd.read_csv(csv_url)
# df = df_raw.copy()
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # --- REFINED PREPROCESSING PIPELINES ---
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# # Define preprocessing pipelines for multi-feature models
# poly_features_list = ["Hours_Studied", "Previous_Scores", "Sleep_Hours"]
#
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('scaler', StandardScaler())
# ])
#
# numeric_scaler = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])
#
# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# # Combine transformers into ColumnTransformers
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('poly_num', numeric_poly_transformer, poly_features_list),
#         ('other_num', numeric_scaler, [col for col in numeric_cols if col not in poly_features_list]),
#         ('cat', categorical_transformer, cat_cols)
#     ],
#     remainder='passthrough'
# )
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', StandardScaler(), numeric_cols),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Define models with robust pipelines
# models = {
#     "Simple Linear Regression": Pipeline(steps=[
#         ('scaler', StandardScaler()),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Linear Regression (Ridge)": Pipeline(steps=[
#         ('preprocessor', preprocessor_linear),
#         ('regressor', Ridge(alpha=1.0))  # Added Ridge regularization with alpha=1.0
#     ]),
#     "Multi-Feature Polynomial (deg=2, Ridge)": Pipeline(steps=[
#         ('preprocessor', preprocessor_poly),
#         ('regressor', Ridge(alpha=1.0))  # Added Ridge regularization with alpha=1.0
#     ])
# }
#
# # Cross-validation results
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         X_cv = X[["Hours_Studied"]]
#     else:
#         X_cv = X
#
#     r2_scores = cross_val_score(model, X_cv, y, cv=5, scoring='r2')
#     mae_scores = -cross_val_score(model, X_cv, y, cv=5, scoring='neg_mean_absolute_error')
#     mse_scores = -cross_val_score(model, X_cv, y, cv=5, scoring='neg_mean_squared_error')
#     rmse_scores = np.sqrt(mse_scores)
#
#     cv_results.append([
#         name,
#         np.mean(r2_scores),
#         np.std(r2_scores),
#         np.mean(mae_scores),
#         np.mean(mse_scores),
#         np.mean(rmse_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV R²", "Std Dev CV R²", "Mean CV MAE", "Mean CV MSE", "Mean CV RMSE"])
# st.dataframe(cv_df.set_index("Model"))
#
# # Train final models and plot Actual vs Predicted
# st.subheader("Actual vs Predicted Scores on Test Set")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))
#
# predictions = {}
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         X_train_specific = X_train[["Hours_Studied"]]
#         X_test_specific = X_test[["Hours_Studied"]]
#     else:
#         X_train_specific = X_train
#         X_test_specific = X_test
#
#     model.fit(X_train_specific, y_train)
#     y_pred = model.predict(X_test_specific)
#     predictions[name] = y_pred
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
#
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, y_pred in predictions.items():
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#
#     test_metrics.append([name, r2, mae, mse, rmse])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.dataframe(test_metrics_df.set_index("Model"))

# ridge tuning
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import warnings
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df_raw = pd.read_csv(csv_url)
# df = df_raw.copy()
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # --- REFINED PREPROCESSING PIPELINES ---
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# poly_features_list = ["Hours_Studied", "Previous_Scores", "Sleep_Hours"]
#
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('scaler', StandardScaler())
# ])
#
# numeric_scaler = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])
#
# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('poly_num', numeric_poly_transformer, poly_features_list),
#         ('other_num', numeric_scaler, [col for col in numeric_cols if col not in poly_features_list]),
#         ('cat', categorical_transformer, cat_cols)
#     ],
#     remainder='passthrough'
# )
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', StandardScaler(), numeric_cols),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # --- HYPERPARAMETER TUNING ---
# # We will use GridSearchCV to find the best alpha for Ridge models.
# st.subheader("Hyperparameter Tuning with GridSearchCV")
#
# # Define a range of alpha values to search
# param_grid = {'regressor__alpha': np.logspace(-4, 4, 10)}
#
# # Define pipelines for the models we want to tune
# multi_linear_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_linear),
#     ('regressor', Ridge())
# ])
#
# poly_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_poly),
#     ('regressor', Ridge())
# ])
#
# # Perform GridSearchCV for each model
# tuned_multi_linear = GridSearchCV(multi_linear_pipeline, param_grid, cv=5, scoring='r2', verbose=1)
# tuned_poly = GridSearchCV(poly_pipeline, param_grid, cv=5, scoring='r2', verbose=1)
#
# # Fit the GridSearchCV models
# tuned_multi_linear.fit(X, y)
# tuned_poly.fit(X, y)
#
# # Get the best parameters and best scores
# best_linear_params = tuned_multi_linear.best_params_
# best_linear_score = tuned_multi_linear.best_score_
# best_poly_params = tuned_poly.best_params_
# best_poly_score = tuned_poly.best_score_
#
# st.write(
#     f"Best Alpha for Multi-Feature Linear Regression (Ridge): **{best_linear_params['regressor__alpha']:.4f}** (R²: {best_linear_score:.4f})")
# st.write(
#     f"Best Alpha for Multi-Feature Polynomial (deg=2, Ridge): **{best_poly_params['regressor__alpha']:.4f}** (R²: {best_poly_score:.4f})")
#
# # Define models with the best hyperparameters
# models = {
#     "Simple Linear Regression": Pipeline(steps=[
#         ('scaler', StandardScaler()),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Linear Regression (Tuned Ridge)": tuned_multi_linear.best_estimator_,
#     "Multi-Feature Polynomial (Tuned Ridge)": tuned_poly.best_estimator_
# }
#
# # Cross-validation results
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         # Note: GridSearchCV already performed cross-validation, so we'll just use its best score.
#         r2_scores = cross_val_score(model, X[["Hours_Studied"]], y, cv=5, scoring='r2')
#         mae_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=5, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=5, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#
#     else:
#         # For the tuned models, their best_estimator_ is already a fitted pipeline,
#         # but we can re-run cross_val_score for a consistent table format.
#         r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
#         mae_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#
#     cv_results.append([
#         name,
#         np.mean(r2_scores),
#         np.std(r2_scores),
#         np.mean(mae_scores),
#         np.mean(mse_scores),
#         np.mean(rmse_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV R²", "Std Dev CV R²", "Mean CV MAE", "Mean CV MSE", "Mean CV RMSE"])
# st.dataframe(cv_df.set_index("Model"))
#
# # Train final models and plot Actual vs Predicted
# st.subheader("Actual vs Predicted Scores on Test Set")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))
#
# predictions = {}
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         X_train_specific = X_train[["Hours_Studied"]]
#         X_test_specific = X_test[["Hours_Studied"]]
#     else:
#         X_train_specific = X_train
#         X_test_specific = X_test
#
#     model.fit(X_train_specific, y_train)
#     y_pred = model.predict(X_test_specific)
#     predictions[name] = y_pred
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
#
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, y_pred in predictions.items():
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#
#     test_metrics.append([name, r2, mae, mse, rmse])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.dataframe(test_metrics_df.set_index("Model"))


# repeat kfold-ANS
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import warnings
from sklearn.impute import SimpleImputer


# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df_raw = pd.read_csv(csv_url)
# df = df_raw.copy()
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Student Score Predictor", layout="wide")
st.title("📊 Student Score Predictor")

# Load dataset
csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
df_raw = pd.read_csv(csv_url)
df = df_raw.copy()

# --- NEW: Advanced Missing Value Imputation ---
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
# ---------------------------------------------------

target = "Exam_Score"

# Define features
features = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
    "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
    "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
    "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
    "Distance_from_Home", "Gender"
]
features = [f for f in features if f in df.columns]

X = df[features]
y = df[target]


# --- REFINED PREPROCESSING PIPELINES ---
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

# --- HYPERPARAMETER TUNING ---
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

# Use RepeatedKFold for a more robust cross-validation within GridSearchCV
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

# Define models with the best hyperparameters
models = {
    "Simple Linear Regression": Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ]),
    "Multi-Feature Linear Regression (Tuned Ridge)": tuned_multi_linear.best_estimator_,
    "Multi-Feature Polynomial (Tuned Ridge)": tuned_poly.best_estimator_
}

# --- CROSS-VALIDATION RESULTS with RepeatedKFold ---
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

# --- VISUALIZATION OF CROSS-VALIDATION RESULTS ---
st.subheader("R² Scores for Each Cross-Validation Fold")
fig_cv = go.Figure()
for name, scores in r2_scores_dict.items():
    fig_cv.add_trace(go.Box(y=scores, name=name))

fig_cv.update_layout(title="R² Scores Distribution across 15 Folds",
                     yaxis_title="R² Score",
                     showlegend=False)
st.plotly_chart(fig_cv, use_container_width=True)

# Train final models and plot Actual vs Predicted
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

# Calculate and show test set performance metrics
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

# hyper tuning
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import PolynomialFeatures, RobustScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import warnings
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df_raw = pd.read_csv(csv_url)
# df = df_raw.copy()
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # --- REFINED PREPROCESSING PIPELINES with RobustScaler ---
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# poly_features_list = ["Hours_Studied", "Previous_Scores", "Sleep_Hours"]
#
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('scaler', RobustScaler())
# ])
#
# numeric_scaler = Pipeline(steps=[
#     ('scaler', RobustScaler())
# ])
#
# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('poly_num', numeric_poly_transformer, poly_features_list),
#         ('other_num', numeric_scaler, [col for col in numeric_cols if col not in poly_features_list]),
#         ('cat', categorical_transformer, cat_cols)
#     ],
#     remainder='passthrough'
# )
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', RobustScaler(), numeric_cols),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # --- HYPERPARAMETER TUNING ---
# st.subheader("Hyperparameter Tuning with GridSearchCV")
#
# param_grid = {'regressor__alpha': np.logspace(-4, 4, 10)}
#
# multi_linear_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_linear),
#     ('regressor', Ridge())
# ])
#
# poly_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_poly),
#     ('regressor', Ridge())
# ])
#
# cv_strategy = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
#
# tuned_multi_linear = GridSearchCV(multi_linear_pipeline, param_grid, cv=cv_strategy, scoring='r2', verbose=1)
# tuned_poly = GridSearchCV(poly_pipeline, param_grid, cv=cv_strategy, scoring='r2', verbose=1)
#
# tuned_multi_linear.fit(X, y)
# tuned_poly.fit(X, y)
#
# best_linear_params = tuned_multi_linear.best_params_
# best_linear_score = tuned_multi_linear.best_score_
# best_poly_params = tuned_poly.best_params_
# best_poly_score = tuned_poly.best_score_
#
# st.write(f"Best Alpha for Multi-Feature Linear Regression (Ridge): **{best_linear_params['regressor__alpha']:.4f}** (R²: {best_linear_score:.4f})")
# st.write(f"Best Alpha for Multi-Feature Polynomial (deg=2, Ridge): **{best_poly_params['regressor__alpha']:.4f}** (R²: {best_poly_score:.4f})")
#
# # Define models with the best hyperparameters
# models = {
#     "Simple Linear Regression": Pipeline(steps=[
#         ('scaler', RobustScaler()),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Linear Regression (Tuned Ridge)": tuned_multi_linear.best_estimator_,
#     "Multi-Feature Polynomial (Tuned Ridge)": tuned_poly.best_estimator_
# }
#
# # --- CROSS-VALIDATION RESULTS with RepeatedKFold ---
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# r2_scores_dict = {}
#
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         r2_scores = cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='r2')
#         mae_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#     else:
#         r2_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2')
#         mae_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#
#     r2_scores_dict[name] = r2_scores
#     cv_results.append([
#         name,
#         np.mean(r2_scores),
#         np.std(r2_scores),
#         np.mean(mae_scores),
#         np.mean(mse_scores),
#         np.mean(rmse_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV R²", "Std Dev CV R²", "Mean CV MAE", "Mean CV MSE", "Mean CV RMSE"])
# st.dataframe(cv_df.set_index("Model"))
#
# # --- VISUALIZATION OF CROSS-VALIDATION RESULTS ---
# st.subheader("R² Scores for Each Cross-Validation Fold")
# fig_cv = go.Figure()
# for name, scores in r2_scores_dict.items():
#     fig_cv.add_trace(go.Box(y=scores, name=name))
#
# fig_cv.update_layout(title="R² Scores Distribution across 15 Folds",
#                      yaxis_title="R² Score",
#                      showlegend=False)
# st.plotly_chart(fig_cv, use_container_width=True)
#
# # Train final models and plot Actual vs Predicted
# st.subheader("Actual vs Predicted Scores on Test Set")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))
#
# predictions = {}
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         X_train_specific = X_train[["Hours_Studied"]]
#         X_test_specific = X_test[["Hours_Studied"]]
#     else:
#         X_train_specific = X_train
#         X_test_specific = X_test
#
#     model.fit(X_train_specific, y_train)
#     y_pred = model.predict(X_test_specific)
#     predictions[name] = y_pred
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
#
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, y_pred in predictions.items():
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#
#     test_metrics.append([name, r2, mae, mse, rmse])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.dataframe(test_metrics_df.set_index("Model"))


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import PolynomialFeatures, RobustScaler, OneHotEncoder, OrdinalEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import warnings
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df_raw = pd.read_csv(csv_url)
# df = df_raw.copy()
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # --- REFINED PREPROCESSING PIPELINES with Ordinal Encoding ---
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# # Define columns for specific transformers
# ordinal_cols = ["Parental_Involvement", "Motivation_Level"]
# onehot_cols = [col for col in cat_cols if col not in ordinal_cols]
# poly_features_list = ["Hours_Studied", "Previous_Scores", "Sleep_Hours", "Attendance"]
#
# # Define the order for ordinal features based on inspection
# ordinal_categories = [
#     ['Low', 'Medium', 'High'], # Parental_Involvement
#     ['Low', 'Medium', 'High']  # Motivation_Level
# ]
#
# # Pipelines for different data types
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=3, include_bias=False)),
#     ('scaler', RobustScaler())
# ])
#
# numeric_scaler = Pipeline(steps=[
#     ('scaler', RobustScaler())
# ])
#
# ordinal_transformer = Pipeline(steps=[
#     ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
# ])
#
# onehot_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# # Combine transformers into ColumnTransformers
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('poly_num', numeric_poly_transformer, poly_features_list),
#         ('other_num', numeric_scaler, [col for col in numeric_cols if col not in poly_features_list]),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', RobustScaler(), numeric_cols),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # --- HYPERPARAMETER TUNING ---
# st.subheader("Hyperparameter Tuning with GridSearchCV")
#
# # Expanded param_grid for polynomial model to tune both alpha and degree
# param_grid_poly = {
#     'preprocessor__poly_num__poly__degree': [1, 2, 3],
#     'regressor__alpha': np.logspace(-4, 4, 10)
# }
# param_grid_linear = {'regressor__alpha': np.logspace(-4, 4, 10)}
#
# multi_linear_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_linear),
#     ('regressor', Ridge())
# ])
#
# poly_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_poly),
#     ('regressor', Ridge())
# ])
#
# cv_strategy = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
#
# tuned_multi_linear = GridSearchCV(multi_linear_pipeline, param_grid_linear, cv=cv_strategy, scoring='r2', verbose=1)
# tuned_poly = GridSearchCV(poly_pipeline, param_grid_poly, cv=cv_strategy, scoring='r2', verbose=1)
#
# tuned_multi_linear.fit(X, y)
# tuned_poly.fit(X, y)
#
# best_linear_params = tuned_multi_linear.best_params_
# best_linear_score = tuned_multi_linear.best_score_
# best_poly_params = tuned_poly.best_params_
# best_poly_score = tuned_poly.best_score_
#
# st.write(f"Best Alpha for Multi-Feature Linear Regression (Ridge): **{best_linear_params.get('regressor__alpha'):.4f}** (R²: {best_linear_score:.4f})")
# st.write(f"Best parameters for Polynomial Model: **{best_poly_params}** (R²: {best_poly_score:.4f})")
#
# # Define models with the best hyperparameters
# models = {
#     "Simple Linear Regression": Pipeline(steps=[
#         ('scaler', RobustScaler()),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Linear Regression (Tuned Ridge)": tuned_multi_linear.best_estimator_,
#     "Multi-Feature Polynomial (Tuned Ridge)": tuned_poly.best_estimator_
# }
#
# # --- CROSS-VALIDATION RESULTS with RepeatedKFold ---
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# r2_scores_dict = {}
#
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         r2_scores = cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='r2')
#         mae_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#     else:
#         r2_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2')
#         mae_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#
#     r2_scores_dict[name] = r2_scores
#     cv_results.append([
#         name,
#         np.mean(r2_scores),
#         np.std(r2_scores),
#         np.mean(mae_scores),
#         np.mean(mse_scores),
#         np.mean(rmse_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV R²", "Std Dev CV R²", "Mean CV MAE", "Mean CV MSE", "Mean CV RMSE"])
# st.dataframe(cv_df.set_index("Model"))
#
# # --- VISUALIZATION OF CROSS-VALIDATION RESULTS ---
# st.subheader("R² Scores for Each Cross-Validation Fold")
# fig_cv = go.Figure()
# for name, scores in r2_scores_dict.items():
#     fig_cv.add_trace(go.Box(y=scores, name=name))
#
# fig_cv.update_layout(title="R² Scores Distribution across 15 Folds",
#                      yaxis_title="R² Score",
#                      showlegend=False)
# st.plotly_chart(fig_cv, use_container_width=True)
#
# # Train final models and plot Actual vs Predicted
# st.subheader("Actual vs Predicted Scores on Test Set")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))
#
# predictions = {}
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         X_train_specific = X_train[["Hours_Studied"]]
#         X_test_specific = X_test[["Hours_Studied"]]
#     else:
#         X_train_specific = X_train
#         X_test_specific = X_test
#
#     model.fit(X_train_specific, y_train)
#     y_pred = model.predict(X_test_specific)
#     predictions[name] = y_pred
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
#
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, y_pred in predictions.items():
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#
#     test_metrics.append([name, r2, mae, mse, rmse])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.dataframe(test_metrics_df.set_index("Model"))


# 5 k fold
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import PolynomialFeatures, RobustScaler, OneHotEncoder, OrdinalEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import warnings
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df_raw = pd.read_csv(csv_url)
# df = df_raw.copy()
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # --- REFINED PREPROCESSING PIPELINES with Ordinal Encoding ---
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# # Define columns for specific transformers
# ordinal_cols = ["Parental_Involvement", "Motivation_Level"]
# onehot_cols = [col for col in cat_cols if col not in ordinal_cols]
# poly_features_list = ["Hours_Studied", "Previous_Scores", "Sleep_Hours", "Attendance"]
#
# # Define the order for ordinal features based on inspection
# ordinal_categories = [
#     ['Low', 'Medium', 'High'], # Parental_Involvement
#     ['Low', 'Medium', 'High']  # Motivation_Level
# ]
#
# # Pipelines for different data types
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=3, include_bias=False)),
#     ('scaler', RobustScaler())
# ])
#
# numeric_scaler = Pipeline(steps=[
#     ('scaler', RobustScaler())
# ])
#
# ordinal_transformer = Pipeline(steps=[
#     ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
# ])
#
# onehot_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# # Combine transformers into ColumnTransformers
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('poly_num', numeric_poly_transformer, poly_features_list),
#         ('other_num', numeric_scaler, [col for col in numeric_cols if col not in poly_features_list]),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', RobustScaler(), numeric_cols),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # --- HYPERPARAMETER TUNING ---
# st.subheader("Hyperparameter Tuning with GridSearchCV")
#
# # Expanded param_grid for polynomial model to tune both alpha and degree
# param_grid_poly = {
#     'preprocessor__poly_num__poly__degree': [1, 2, 3],
#     'regressor__alpha': np.logspace(-4, 4, 10)
# }
# param_grid_linear = {'regressor__alpha': np.logspace(-4, 4, 10)}
#
# multi_linear_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_linear),
#     ('regressor', Ridge())
# ])
#
# poly_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_poly),
#     ('regressor', Ridge())
# ])
#
# # Changed from RepeatedKFold to KFold with 5 splits
# cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
#
# tuned_multi_linear = GridSearchCV(multi_linear_pipeline, param_grid_linear, cv=cv_strategy, scoring='r2', verbose=1)
# tuned_poly = GridSearchCV(poly_pipeline, param_grid_poly, cv=cv_strategy, scoring='r2', verbose=1)
#
# tuned_multi_linear.fit(X, y)
# tuned_poly.fit(X, y)
#
# best_linear_params = tuned_multi_linear.best_params_
# best_linear_score = tuned_multi_linear.best_score_
# best_poly_params = tuned_poly.best_params_
# best_poly_score = tuned_poly.best_score_
#
# st.write(f"Best Alpha for Multi-Feature Linear Regression (Ridge): **{best_linear_params.get('regressor__alpha'):.4f}** (R²: {best_linear_score:.4f})")
# st.write(f"Best parameters for Polynomial Model: **{best_poly_params}** (R²: {best_poly_score:.4f})")
#
# # Define models with the best hyperparameters
# models = {
#     "Simple Linear Regression": Pipeline(steps=[
#         ('scaler', RobustScaler()),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Linear Regression (Tuned Ridge)": tuned_multi_linear.best_estimator_,
#     "Multi-Feature Polynomial (Tuned Ridge)": tuned_poly.best_estimator_
# }
#
# # --- CROSS-VALIDATION RESULTS with KFold ---
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# r2_scores_dict = {}
#
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         r2_scores = cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='r2')
#         mae_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#     else:
#         r2_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2')
#         mae_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#
#     r2_scores_dict[name] = r2_scores
#     cv_results.append([
#         name,
#         np.mean(r2_scores),
#         np.std(r2_scores),
#         np.mean(mae_scores),
#         np.mean(mse_scores),
#         np.mean(rmse_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV R²", "Std Dev CV R²", "Mean CV MAE", "Mean CV MSE", "Mean CV RMSE"])
# st.dataframe(cv_df.set_index("Model"))
#
# # --- VISUALIZATION OF CROSS-VALIDATION RESULTS ---
# st.subheader("R² Scores for Each Cross-Validation Fold")
# fig_cv = go.Figure()
# for name, scores in r2_scores_dict.items():
#     fig_cv.add_trace(go.Box(y=scores, name=name))
#
# fig_cv.update_layout(title="R² Scores Distribution across 5 Folds",
#                      yaxis_title="R² Score",
#                      showlegend=False)
# st.plotly_chart(fig_cv, use_container_width=True)
#
# # Train final models and plot Actual vs Predicted
# st.subheader("Actual vs Predicted Scores on Test Set")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))
#
# predictions = {}
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         X_train_specific = X_train[["Hours_Studied"]]
#         X_test_specific = X_test[["Hours_Studied"]]
#     else:
#         X_train_specific = X_train
#         X_test_specific = X_test
#
#     model.fit(X_train_specific, y_train)
#     y_pred = model.predict(X_test_specific)
#     predictions[name] = y_pred
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
#
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, y_pred in predictions.items():
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#
#     test_metrics.append([name, r2, mae, mse, rmse])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.dataframe(test_metrics_df.set_index("Model"))


# classification
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import PolynomialFeatures, RobustScaler, OneHotEncoder, OrdinalEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import accuracy_score, classification_report
# import plotly.graph_objects as go
# import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor - Grade Classification")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df_raw = pd.read_csv(csv_url)
# df = df_raw.copy()
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
#
# # Function to convert scores to grade categories
# def convert_to_grade(score):
#     if score >= 90:
#         return 'Excellent'
#     elif score >= 80:
#         return 'Good'
#     elif score >= 70:
#         return 'Average'
#     else:
#         return 'Poor'
#
#
# # Apply the conversion to create a new target column
# df['Grade'] = df['Exam_Score'].apply(convert_to_grade)
#
# target = "Grade"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # --- REFINED PREPROCESSING PIPELINES with Ordinal Encoding ---
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# # Define columns for specific transformers
# ordinal_cols = ["Parental_Involvement", "Motivation_Level"]
# onehot_cols = [col for col in cat_cols if col not in ordinal_cols]
# poly_features_list = ["Hours_Studied", "Previous_Scores", "Sleep_Hours", "Attendance"]
#
# # Define the order for ordinal features based on inspection
# ordinal_categories = [
#     ['Low', 'Medium', 'High'],  # Parental_Involvement
#     ['Low', 'Medium', 'High']  # Motivation_Level
# ]
#
# # Pipelines for different data types
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('scaler', RobustScaler())
# ])
#
# numeric_scaler = Pipeline(steps=[
#     ('scaler', RobustScaler())
# ])
#
# ordinal_transformer = Pipeline(steps=[
#     ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
# ])
#
# onehot_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# # Combine transformers into ColumnTransformers
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('poly_num', numeric_poly_transformer, poly_features_list),
#         ('other_num', numeric_scaler, [col for col in numeric_cols if col not in poly_features_list]),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', RobustScaler(), numeric_cols),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # --- HYPERPARAMETER TUNING ---
# st.subheader("Hyperparameter Tuning with GridSearchCV")
#
# # Parameter grids for classification models
# param_grid_logreg = {'classifier__C': np.logspace(-4, 4, 10)}
# param_grid_rf = {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [3, 5, 10]}
#
# logreg_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_linear),
#     ('classifier', LogisticRegression(solver='liblinear', random_state=42))
# ])
#
# rf_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_poly),
#     ('classifier', RandomForestClassifier(random_state=42))
# ])
#
# cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
#
# tuned_logreg = GridSearchCV(logreg_pipeline, param_grid_logreg, cv=cv_strategy, scoring='accuracy', verbose=1)
# tuned_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=cv_strategy, scoring='accuracy', verbose=1)
#
# tuned_logreg.fit(X, y)
# tuned_rf.fit(X, y)
#
# best_logreg_params = tuned_logreg.best_params_
# best_logreg_score = tuned_logreg.best_score_
# best_rf_params = tuned_rf.best_params_
# best_rf_score = tuned_rf.best_score_
#
# st.write(
#     f"Best C for Logistic Regression: **{best_logreg_params.get('classifier__C'):.4f}** (Accuracy: {best_logreg_score:.4f})")
# st.write(f"Best parameters for Random Forest: **{best_rf_params}** (Accuracy: {best_rf_score:.4f})")
#
# # Define models with the best hyperparameters
# models = {
#     "Logistic Regression (Tuned)": tuned_logreg.best_estimator_,
#     "Random Forest (Tuned)": tuned_rf.best_estimator_
# }
#
# # --- CROSS-VALIDATION RESULTS with KFold ---
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# accuracy_scores_dict = {}
#
# for name, model in models.items():
#     accuracy_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy')
#
#     accuracy_scores_dict[name] = accuracy_scores
#     cv_results.append([
#         name,
#         np.mean(accuracy_scores),
#         np.std(accuracy_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV Accuracy", "Std Dev CV Accuracy"])
# st.dataframe(cv_df.set_index("Model"))
#
# # --- VISUALIZATION OF CROSS-VALIDATION RESULTS ---
# st.subheader("Accuracy Scores for Each Cross-Validation Fold")
# fig_cv = go.Figure()
# for name, scores in accuracy_scores_dict.items():
#     fig_cv.add_trace(go.Box(y=scores, name=name))
#
# fig_cv.update_layout(title="Accuracy Scores Distribution across 5 Folds",
#                      yaxis_title="Accuracy Score",
#                      showlegend=False)
# st.plotly_chart(fig_cv, use_container_width=True)
#
# # Train final models and plot Actual vs Predicted Grade Counts
# st.subheader("Actual vs Predicted Grade Counts on Test Set")
#
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#
# y_test_counts = y_test.value_counts().sort_index()
# sns.barplot(x=y_test_counts.index, y=y_test_counts.values, ax=axes[0], palette="viridis")
# axes[0].set_title('Actual Grade Counts')
# axes[0].set_xlabel('Grade')
# axes[0].set_ylabel('Count')
#
# predictions = {}
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     predictions[name] = y_pred
#
#     y_pred_counts = pd.Series(y_pred).value_counts().sort_index()
#     sns.barplot(x=y_pred_counts.index, y=y_pred_counts.values, ax=axes[1], palette="plasma")
#     axes[1].set_title(f'Predicted Grade Counts ({name})')
#     axes[1].set_xlabel('Grade')
#     axes[1].set_ylabel('Count')
#
# st.pyplot(fig)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, y_pred in predictions.items():
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred, output_dict=True)
#
#     # Extract key metrics from the classification report
#     precision = report['weighted avg']['precision']
#     recall = report['weighted avg']['recall']
#     f1_score = report['weighted avg']['f1-score']
#
#     test_metrics.append([name, accuracy, precision, recall, f1_score])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "Accuracy", "Weighted Precision", "Weighted Recall",
#                                         "Weighted F1-Score"])
# st.dataframe(test_metrics_df.set_index("Model"))

#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import PolynomialFeatures, RobustScaler, OneHotEncoder, OrdinalEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import plotly.graph_objects as go
# import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor - Grade Classification")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df_raw = pd.read_csv(csv_url)
# df = df_raw.copy()
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
#
# # Function to convert scores to grade categories
# def convert_to_grade(score):
#     if score >= 90:
#         return 'Excellent'
#     elif score >= 80:
#         return 'Good'
#     elif score >= 70:
#         return 'Average'
#     else:
#         return 'Poor'
#
#
# # Apply the conversion to create a new target column
# df['Grade'] = df['Exam_Score'].apply(convert_to_grade)
#
# target = "Grade"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # --- REFINED PREPROCESSING PIPELINES with Ordinal Encoding ---
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# # Define columns for specific transformers
# ordinal_cols = ["Parental_Involvement", "Motivation_Level"]
# onehot_cols = [col for col in cat_cols if col not in ordinal_cols]
# poly_features_list = ["Hours_Studied", "Previous_Scores", "Sleep_Hours", "Attendance"]
#
# # Define the order for ordinal features based on inspection
# ordinal_categories = [
#     ['Low', 'Medium', 'High'],  # Parental_Involvement
#     ['Low', 'Medium', 'High']  # Motivation_Level
# ]
#
# # Pipelines for different data types
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('scaler', RobustScaler())
# ])
#
# numeric_scaler = Pipeline(steps=[
#     ('scaler', RobustScaler())
# ])
#
# ordinal_transformer = Pipeline(steps=[
#     ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
# ])
#
# onehot_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# # Combine transformers into ColumnTransformers
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('poly_num', numeric_poly_transformer, poly_features_list),
#         ('other_num', numeric_scaler, [col for col in numeric_cols if col not in poly_features_list]),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', RobustScaler(), numeric_cols),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # --- EXPLORATORY DATA ANALYSIS VISUALIZATIONS ---
# st.subheader("Data Understanding and Visualizations")
#
# st.write("### Correlation Heatmap of Numeric Features")
# df_numeric_corr = df[numeric_cols + ['Exam_Score']].copy()
# # Map grades to numbers for correlation
# df_numeric_corr['Grade_num'] = df['Grade'].map({'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3})
# corr_matrix = df_numeric_corr.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# st.pyplot(plt)
#
# st.write("### Distribution of Categorical Features")
# cat_display_cols = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level',
#                     'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence',
#                     'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender']
# num_plots = len(cat_display_cols)
# num_cols_per_row = 3
# num_rows = int(np.ceil(num_plots / num_cols_per_row))
# fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=(18, 5 * num_rows))
# axes = axes.flatten()
#
# for i, col in enumerate(cat_display_cols):
#     sns.countplot(y=df[col], ax=axes[i], palette='viridis')
#     axes[i].set_title(f'Distribution of {col}')
#     axes[i].set_xlabel('Count')
#     axes[i].set_ylabel('')
#
# # Hide any unused subplots
# for j in range(i + 1, len(axes)):
#     axes[j].axis('off')
#
# plt.tight_layout()
# st.pyplot(fig)
#
# # --- HYPERPARAMETER TUNING ---
# st.subheader("Hyperparameter Tuning with GridSearchCV")
#
# # Parameter grids for classification models
# param_grid_logreg = {'classifier__C': np.logspace(-4, 4, 10)}
# param_grid_rf = {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [3, 5, 10]}
#
# logreg_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_linear),
#     ('classifier', LogisticRegression(solver='liblinear', random_state=42))
# ])
#
# rf_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_poly),
#     ('classifier', RandomForestClassifier(random_state=42))
# ])
#
# cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
#
# tuned_logreg = GridSearchCV(logreg_pipeline, param_grid_logreg, cv=cv_strategy, scoring='accuracy', verbose=1)
# tuned_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=cv_strategy, scoring='accuracy', verbose=1)
#
# tuned_logreg.fit(X, y)
# tuned_rf.fit(X, y)
#
# best_logreg_params = tuned_logreg.best_params_
# best_logreg_score = tuned_logreg.best_score_
# best_rf_params = tuned_rf.best_params_
# best_rf_score = tuned_rf.best_score_
#
# st.write(
#     f"Best C for Logistic Regression: **{best_logreg_params.get('classifier__C'):.4f}** (Accuracy: {best_logreg_score:.4f})")
# st.write(f"Best parameters for Random Forest: **{best_rf_params}** (Accuracy: {best_rf_score:.4f})")
#
# # Define models with the best hyperparameters
# models = {
#     "Logistic Regression (Tuned)": tuned_logreg.best_estimator_,
#     "Random Forest (Tuned)": tuned_rf.best_estimator_
# }
#
# # --- CROSS-VALIDATION RESULTS with KFold ---
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# accuracy_scores_dict = {}
#
# for name, model in models.items():
#     accuracy_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy')
#
#     accuracy_scores_dict[name] = accuracy_scores
#     cv_results.append([
#         name,
#         np.mean(accuracy_scores),
#         np.std(accuracy_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV Accuracy", "Std Dev CV Accuracy"])
# st.dataframe(cv_df.set_index("Model"))
#
# # --- VISUALIZATION OF CROSS-VALIDATION RESULTS ---
# st.subheader("Accuracy Scores for Each Cross-Validation Fold")
# fig_cv = go.Figure()
# for name, scores in accuracy_scores_dict.items():
#     fig_cv.add_trace(go.Box(y=scores, name=name))
#
# fig_cv.update_layout(title="Accuracy Scores Distribution across 5 Folds",
#                      yaxis_title="Accuracy Score",
#                      showlegend=False)
# st.plotly_chart(fig_cv, use_container_width=True)
#
# # Train final models and plot Actual vs Predicted Grade Counts
# st.subheader("Actual vs Predicted Grade Counts on Test Set")
#
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#
# y_test_counts = y_test.value_counts().sort_index()
# sns.barplot(x=y_test_counts.index, y=y_test_counts.values, ax=axes[0], palette="viridis")
# axes[0].set_title('Actual Grade Counts')
# axes[0].set_xlabel('Grade')
# axes[0].set_ylabel('Count')
#
# predictions = {}
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     predictions[name] = y_pred
#
#     y_pred_counts = pd.Series(y_pred).value_counts().sort_index()
#     sns.barplot(x=y_pred_counts.index, y=y_pred_counts.values, ax=axes[1], palette="plasma")
#     axes[1].set_title(f'Predicted Grade Counts ({name})')
#     axes[1].set_xlabel('Grade')
#     axes[1].set_ylabel('Count')
#
# st.pyplot(fig)
#
# # --- NEW VISUALIZATIONS FOR MODEL PERFORMANCE AND EXPLAINABILITY ---
# st.subheader("Detailed Model Performance Visualizations")
#
# # Confusion Matrices
# st.write("### Confusion Matrices")
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# grade_order = ['Poor', 'Average', 'Good', 'Excellent']
#
# for i, (name, y_pred) in enumerate(predictions.items()):
#     cm = confusion_matrix(y_test, y_pred, labels=grade_order)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
#     axes[i].set_title(f'Confusion Matrix - {name}')
#     axes[i].set_xlabel('Predicted Grade')
#     axes[i].set_ylabel('Actual Grade')
#     axes[i].set_xticklabels(grade_order, rotation=45, ha='right')
#     axes[i].set_yticklabels(grade_order, rotation=0, ha='right')
#
# plt.tight_layout()
# st.pyplot(fig)
#
# # Feature Importance (for Random Forest)
# st.write("### Feature Importance (Random Forest)")
# rf_model = models["Random Forest (Tuned)"]
# # Get feature names from the preprocessor
# feature_names = rf_model.named_steps['preprocessor'].get_feature_names_out()
# importances = rf_model.named_steps['classifier'].feature_importances_
# feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance',
#                                                                                                       ascending=False)
#
# plt.figure(figsize=(10, 8))
# sns.barplot(x='importance', y='feature', data=feature_importances.head(20), palette='viridis')
# plt.title('Top 20 Feature Importances')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.tight_layout()
# st.pyplot(plt)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, y_pred in predictions.items():
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred, output_dict=True)
#
#     # Extract key metrics from the classification report
#     precision = report['weighted avg']['precision']
#     recall = report['weighted avg']['recall']
#     f1_score = report['weighted avg']['f1-score']
#
#     test_metrics.append([name, accuracy, precision, recall, f1_score])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "Accuracy", "Weighted Precision", "Weighted Recall",
#                                         "Weighted F1-Score"])
# st.dataframe(test_metrics_df.set_index("Model"))

#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import PolynomialFeatures, RobustScaler, OneHotEncoder, OrdinalEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor - Regression Analysis")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df_raw = pd.read_csv(csv_url)
# df = df_raw.copy()
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # --- REFINED PREPROCESSING PIPELINES with Ordinal Encoding ---
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# # Define columns for specific transformers
# ordinal_cols = ["Parental_Involvement", "Motivation_Level"]
# onehot_cols = [col for col in cat_cols if col not in ordinal_cols]
# poly_features_list = ["Hours_Studied", "Previous_Scores", "Sleep_Hours", "Attendance"]
#
# # Define the order for ordinal features based on inspection
# ordinal_categories = [
#     ['Low', 'Medium', 'High'], # Parental_Involvement
#     ['Low', 'Medium', 'High']  # Motivation_Level
# ]
#
# # Pipelines for different data types
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('scaler', RobustScaler())
# ])
#
# numeric_scaler = Pipeline(steps=[
#     ('scaler', RobustScaler())
# ])
#
# ordinal_transformer = Pipeline(steps=[
#     ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
# ])
#
# onehot_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# # Combine transformers into ColumnTransformers
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('poly_num', numeric_poly_transformer, poly_features_list),
#         ('other_num', numeric_scaler, [col for col in numeric_cols if col not in poly_features_list]),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', RobustScaler(), numeric_cols),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # --- EXPLORATORY DATA ANALYSIS VISUALIZATIONS ---
# st.subheader("Data Understanding and Visualizations")
#
# st.write("### Correlation Heatmap of Numeric Features")
# df_numeric_corr = df[numeric_cols + ['Exam_Score']].copy()
# corr_matrix = df_numeric_corr.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# st.pyplot(plt)
#
# st.write("### Distribution of Categorical Features")
# cat_display_cols = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender']
# num_plots = len(cat_display_cols)
# num_cols_per_row = 3
# num_rows = int(np.ceil(num_plots / num_cols_per_row))
# fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=(18, 5 * num_rows))
# axes = axes.flatten()
#
# for i, col in enumerate(cat_display_cols):
#     sns.countplot(y=df[col], ax=axes[i], palette='viridis')
#     axes[i].set_title(f'Distribution of {col}')
#     axes[i].set_xlabel('Count')
#     axes[i].set_ylabel('')
#
# # Hide any unused subplots
# for j in range(i + 1, len(axes)):
#     axes[j].axis('off')
#
# plt.tight_layout()
# st.pyplot(fig)
#
# # --- HYPERPARAMETER TUNING ---
# st.subheader("Hyperparameter Tuning with GridSearchCV")
#
# # Parameter grids for regression models
# param_grid_poly = {
#     'preprocessor__poly_num__poly__degree': [1, 2, 3],
#     'regressor__alpha': np.logspace(-4, 4, 10)
# }
# param_grid_linear = {'regressor__alpha': np.logspace(-4, 4, 10)}
#
# multi_linear_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_linear),
#     ('regressor', Ridge())
# ])
#
# poly_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_poly),
#     ('regressor', Ridge())
# ])
#
# cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
#
# tuned_multi_linear = GridSearchCV(multi_linear_pipeline, param_grid_linear, cv=cv_strategy, scoring='r2', verbose=1)
# tuned_poly = GridSearchCV(poly_pipeline, param_grid_poly, cv=cv_strategy, scoring='r2', verbose=1)
#
# tuned_multi_linear.fit(X, y)
# tuned_poly.fit(X, y)
#
# best_linear_params = tuned_multi_linear.best_params_
# best_linear_score = tuned_multi_linear.best_score_
# best_poly_params = tuned_poly.best_params_
# best_poly_score = tuned_poly.best_score_
#
# st.write(f"Best Alpha for Multi-Feature Linear Regression (Ridge): **{best_linear_params.get('regressor__alpha'):.4f}** (R²: {best_linear_score:.4f})")
# st.write(f"Best parameters for Polynomial Model: **{best_poly_params}** (R²: {best_poly_score:.4f})")
#
# # Define models with the best hyperparameters
# models = {
#     "Simple Linear Regression": Pipeline(steps=[
#         ('scaler', RobustScaler()),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Linear Regression (Tuned Ridge)": tuned_multi_linear.best_estimator_,
#     "Multi-Feature Polynomial (Tuned Ridge)": tuned_poly.best_estimator_
# }
#
# # --- CROSS-VALIDATION RESULTS with KFold ---
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# r2_scores_dict = {}
#
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         r2_scores = cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='r2')
#         mae_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#     else:
#         r2_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2')
#         mae_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#
#     r2_scores_dict[name] = r2_scores
#     cv_results.append([
#         name,
#         np.mean(r2_scores),
#         np.std(r2_scores),
#         np.mean(mae_scores),
#         np.mean(mse_scores),
#         np.mean(rmse_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV R²", "Std Dev CV R²", "Mean CV MAE", "Mean CV MSE", "Mean CV RMSE"])
# st.dataframe(cv_df.set_index("Model"))
#
# # --- VISUALIZATION OF CROSS-VALIDATION RESULTS ---
# st.subheader("R² Scores for Each Cross-Validation Fold")
# fig_cv = go.Figure()
# for name, scores in r2_scores_dict.items():
#     fig_cv.add_trace(go.Box(y=scores, name=name))
#
# fig_cv.update_layout(title="R² Scores Distribution across 5 Folds",
#                      yaxis_title="R² Score",
#                      showlegend=False)
# st.plotly_chart(fig_cv, use_container_width=True)
#
# # Train final models and plot Actual vs Predicted
# st.subheader("Actual vs Predicted Scores on Test Set")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white")))
#
# predictions = {}
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         X_train_specific = X_train[["Hours_Studied"]]
#         X_test_specific = X_test[["Hours_Studied"]]
#     else:
#         X_train_specific = X_train
#         X_test_specific = X_test
#
#     model.fit(X_train_specific, y_train)
#     y_pred = model.predict(X_test_specific)
#     predictions[name] = y_pred
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
#
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, y_pred in predictions.items():
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#
#     test_metrics.append([name, r2, mae, mse, rmse])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.dataframe(test_metrics_df.set_index("Model"))
#


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import PolynomialFeatures, RobustScaler, OneHotEncoder, OrdinalEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import plotly.express as px
# import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor - Regression Analysis")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df_raw = pd.read_csv(csv_url)
# df = df_raw.copy()
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# target = "Exam_Score"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # --- REFINED PREPROCESSING PIPELINES with Ordinal Encoding ---
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# # Define columns for specific transformers
# ordinal_cols = ["Parental_Involvement", "Motivation_Level"]
# onehot_cols = [col for col in cat_cols if col not in ordinal_cols]
# poly_features_list = ["Hours_Studied", "Previous_Scores", "Sleep_Hours", "Attendance"]
#
# # Define the order for ordinal features based on inspection
# ordinal_categories = [
#     ['Low', 'Medium', 'High'], # Parental_Involvement
#     ['Low', 'Medium', 'High']  # Motivation_Level
# ]
#
# # Pipelines for different data types
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('scaler', RobustScaler())
# ])
#
# numeric_scaler = Pipeline(steps=[
#     ('scaler', RobustScaler())
# ])
#
# ordinal_transformer = Pipeline(steps=[
#     ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
# ])
#
# onehot_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# # Combine transformers into ColumnTransformers
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('poly_num', numeric_poly_transformer, poly_features_list),
#         ('other_num', numeric_scaler, [col for col in numeric_cols if col not in poly_features_list]),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', RobustScaler(), numeric_cols),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # --- EXPLORATORY DATA ANALYSIS VISUALIZATIONS ---
# st.subheader("Data Understanding and Visualizations")
#
# st.write("### Interactive Correlation Heatmap of Numeric Features")
# df_numeric_corr = df[numeric_cols + ['Exam_Score']].copy()
# corr_matrix = df_numeric_corr.corr()
# fig = go.Figure(data=go.Heatmap(
#     z=corr_matrix.values,
#     x=corr_matrix.columns,
#     y=corr_matrix.index,
#     colorscale='Viridis',
#     hoverongaps=False))
# fig.update_layout(height=600, width=800, title='Correlation Matrix')
# st.plotly_chart(fig, use_container_width=True)
#
# st.write("### Interactive Distribution of Categorical Features")
# cat_display_cols = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender']
# selected_cat_col = st.selectbox("Select a categorical feature to visualize:", cat_display_cols)
# fig = px.histogram(df, x=selected_cat_col, color=selected_cat_col, title=f'Distribution of {selected_cat_col}')
# st.plotly_chart(fig, use_container_width=True)
#
#
# # --- HYPERPARAMETER TUNING ---
# st.subheader("Hyperparameter Tuning with GridSearchCV")
#
# # Parameter grids for regression models
# param_grid_poly = {
#     'preprocessor__poly_num__poly__degree': [1, 2, 3],
#     'regressor__alpha': np.logspace(-4, 4, 10)
# }
# param_grid_linear = {'regressor__alpha': np.logspace(-4, 4, 10)}
#
# multi_linear_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_linear),
#     ('regressor', Ridge())
# ])
#
# poly_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_poly),
#     ('regressor', Ridge())
# ])
#
# cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
#
# tuned_multi_linear = GridSearchCV(multi_linear_pipeline, param_grid_linear, cv=cv_strategy, scoring='r2', verbose=1)
# tuned_poly = GridSearchCV(poly_pipeline, param_grid_poly, cv=cv_strategy, scoring='r2', verbose=1)
#
# tuned_multi_linear.fit(X, y)
# tuned_poly.fit(X, y)
#
# best_linear_params = tuned_multi_linear.best_params_
# best_linear_score = tuned_multi_linear.best_score_
# best_poly_params = tuned_poly.best_params_
# best_poly_score = tuned_poly.best_score_
#
# st.write(f"Best Alpha for Multi-Feature Linear Regression (Ridge): **{best_linear_params.get('regressor__alpha'):.4f}** (R²: {best_linear_score:.4f})")
# st.write(f"Best parameters for Polynomial Model: **{best_poly_params}** (R²: {best_poly_score:.4f})")
#
# # Define models with the best hyperparameters
# models = {
#     "Simple Linear Regression": Pipeline(steps=[
#         ('scaler', RobustScaler()),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Linear Regression (Tuned Ridge)": tuned_multi_linear.best_estimator_,
#     "Multi-Feature Polynomial (Tuned Ridge)": tuned_poly.best_estimator_
# }
#
# # --- CROSS-VALIDATION RESULTS with KFold ---
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# r2_scores_dict = {}
#
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         r2_scores = cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='r2')
#         mae_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#     else:
#         r2_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2')
#         mae_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#
#     r2_scores_dict[name] = r2_scores
#     cv_results.append([
#         name,
#         np.mean(r2_scores),
#         np.std(r2_scores),
#         np.mean(mae_scores),
#         np.mean(mse_scores),
#         np.mean(rmse_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV R²", "Std Dev CV R²", "Mean CV MAE", "Mean CV MSE", "Mean CV RMSE"])
# st.dataframe(cv_df.set_index("Model"))
#
# # --- VISUALIZATION OF CROSS-VALIDATION RESULTS ---
# st.subheader("R² Scores for Each Cross-Validation Fold")
# fig_cv = go.Figure()
# for name, scores in r2_scores_dict.items():
#     fig_cv.add_trace(go.Box(y=scores, name=name))
#
# fig_cv.update_layout(title="R² Scores Distribution across 5 Folds",
#                      yaxis_title="R² Score",
#                      showlegend=False)
# st.plotly_chart(fig_cv, use_container_width=True)
#
# # Train final models and plot Actual vs Predicted
# st.subheader("Actual vs Predicted Scores on Test Set")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white", dash='dot')))
#
# predictions = {}
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         X_train_specific = X_train[["Hours_Studied"]]
#         X_test_specific = X_test[["Hours_Studied"]]
#     else:
#         X_train_specific = X_train
#         X_test_specific = X_test
#
#     model.fit(X_train_specific, y_train)
#     y_pred = model.predict(X_test_specific)
#     predictions[name] = y_pred
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
#
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, y_pred in predictions.items():
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#
#     test_metrics.append([name, r2, mae, mse, rmse])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.dataframe(test_metrics_df.set_index("Model"))


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import PolynomialFeatures, RobustScaler, OneHotEncoder, OrdinalEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import plotly.express as px
# import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Suppress the UserWarning from ColumnTransformer when a list is empty
# warnings.filterwarnings("ignore", category=UserWarning)
#
# st.set_page_config(page_title="Student Score Predictor", layout="wide")
# st.title("📊 Student Score Predictor - Regression Analysis")
#
# # Load dataset
# csv_url = "https://raw.githubusercontent.com/Rasheeq28/datasets/main/StudentPerformanceFactors.csv"
# df_raw = pd.read_csv(csv_url)
# df = df_raw.copy()
#
# # Fill missing values with mode (most frequent) for each column
# for col in df.columns:
#     mode_val = df[col].mode()
#     if not mode_val.empty:
#         df[col].fillna(mode_val[0], inplace=True)
#     else:
#         df[col].fillna(method='ffill', inplace=True)
#
# # --- OUTLIER REMOVAL (IQR Method) ---
# st.subheader("Outlier Removal")
# st.write("Removing outliers from numerical columns using the Interquartile Range (IQR) method.")
#
# def remove_outliers_iqr(df, columns):
#     for col in columns:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#     return df
#
# # Get a fresh list of numerical columns for outlier removal
# numeric_cols_for_outliers = df.select_dtypes(include=np.number).columns.tolist()
#
# # The target variable 'Exam_Score' should also be cleaned
# df_cleaned = remove_outliers_iqr(df, numeric_cols_for_outliers)
#
# st.write(f"Original dataset size: {len(df)} rows")
# st.write(f"Dataset size after outlier removal: {len(df_cleaned)} rows")
#
# df = df_cleaned.copy()
#
# target = "Exam_Score"
#
# # Define features
# features = [
#     "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
#     "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
#     "Motivation_Level", "Internet_Access", "Tutoring_Sessions",
#     "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
#     "Physical_Activity", "Learning_Disabilities", "Parental_Education_Level",
#     "Distance_from_Home", "Gender"
# ]
# features = [f for f in features if f in df.columns]
#
# X = df[features]
# y = df[target]
#
# # --- REFINED PREPROCESSING PIPELINES with Ordinal Encoding ---
# numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#
# # Define columns for specific transformers
# ordinal_cols = ["Parental_Involvement", "Motivation_Level"]
# onehot_cols = [col for col in cat_cols if col not in ordinal_cols]
# poly_features_list = ["Hours_Studied", "Previous_Scores", "Sleep_Hours", "Attendance"]
#
# # Define the order for ordinal features based on inspection
# ordinal_categories = [
#     ['Low', 'Medium', 'High'], # Parental_Involvement
#     ['Low', 'Medium', 'High']  # Motivation_Level
# ]
#
# # Pipelines for different data types
# numeric_poly_transformer = Pipeline(steps=[
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('scaler', RobustScaler())
# ])
#
# numeric_scaler = Pipeline(steps=[
#     ('scaler', RobustScaler())
# ])
#
# ordinal_transformer = Pipeline(steps=[
#     ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
# ])
#
# onehot_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# # Combine transformers into ColumnTransformers
# preprocessor_poly = ColumnTransformer(
#     transformers=[
#         ('poly_num', numeric_poly_transformer, poly_features_list),
#         ('other_num', numeric_scaler, [col for col in numeric_cols if col not in poly_features_list]),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ('scaler', RobustScaler(), numeric_cols),
#         ('ordinal', ordinal_transformer, ordinal_cols),
#         ('onehot', onehot_transformer, onehot_cols)
#     ],
#     remainder='passthrough'
# )
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # --- EXPLORATORY DATA ANALYSIS VISUALIZATIONS ---
# st.subheader("Data Understanding and Visualizations")
#
# st.write("### Interactive Correlation Heatmap of Numeric Features")
# df_numeric_corr = df[numeric_cols + ['Exam_Score']].copy()
# corr_matrix = df_numeric_corr.corr()
# fig = go.Figure(data=go.Heatmap(
#     z=corr_matrix.values,
#     x=corr_matrix.columns,
#     y=corr_matrix.index,
#     colorscale='Viridis',
#     hoverongaps=False))
# fig.update_layout(height=600, width=800, title='Correlation Matrix')
# st.plotly_chart(fig, use_container_width=True)
#
# st.write("### Interactive Distribution of Categorical Features")
# cat_display_cols = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender']
# selected_cat_col = st.selectbox("Select a categorical feature to visualize:", cat_display_cols)
# fig = px.histogram(df, x=selected_cat_col, color=selected_cat_col, title=f'Distribution of {selected_cat_col}')
# st.plotly_chart(fig, use_container_width=True)
#
#
# # --- HYPERPARAMETER TUNING ---
# st.subheader("Hyperparameter Tuning with GridSearchCV")
#
# # Parameter grids for regression models
# param_grid_poly = {
#     'preprocessor__poly_num__poly__degree': [1, 2, 3],
#     'regressor__alpha': np.logspace(-4, 4, 10)
# }
# param_grid_linear = {'regressor__alpha': np.logspace(-4, 4, 10)}
#
# multi_linear_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_linear),
#     ('regressor', Ridge())
# ])
#
# poly_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor_poly),
#     ('regressor', Ridge())
# ])
#
# cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
#
# tuned_multi_linear = GridSearchCV(multi_linear_pipeline, param_grid_linear, cv=cv_strategy, scoring='r2', verbose=1)
# tuned_poly = GridSearchCV(poly_pipeline, param_grid_poly, cv=cv_strategy, scoring='r2', verbose=1)
#
# tuned_multi_linear.fit(X, y)
# tuned_poly.fit(X, y)
#
# best_linear_params = tuned_multi_linear.best_params_
# best_linear_score = tuned_multi_linear.best_score_
# best_poly_params = tuned_poly.best_params_
# best_poly_score = tuned_poly.best_score_
#
# st.write(f"Best Alpha for Multi-Feature Linear Regression (Ridge): **{best_linear_params.get('regressor__alpha'):.4f}** (R²: {best_linear_score:.4f})")
# st.write(f"Best parameters for Polynomial Model: **{best_poly_params}** (R²: {best_poly_score:.4f})")
#
# # Define models with the best hyperparameters
# models = {
#     "Simple Linear Regression": Pipeline(steps=[
#         ('scaler', RobustScaler()),
#         ('regressor', LinearRegression())
#     ]),
#     "Multi-Feature Linear Regression (Tuned Ridge)": tuned_multi_linear.best_estimator_,
#     "Multi-Feature Polynomial (Tuned Ridge)": tuned_poly.best_estimator_
# }
#
# # --- CROSS-VALIDATION RESULTS with KFold ---
# st.subheader("Model Performance with Cross-Validation")
# cv_results = []
# r2_scores_dict = {}
#
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         r2_scores = cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='r2')
#         mae_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X[["Hours_Studied"]], y, cv=cv_strategy, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#     else:
#         r2_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2')
#         mae_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_absolute_error')
#         mse_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_squared_error')
#         rmse_scores = np.sqrt(mse_scores)
#
#     r2_scores_dict[name] = r2_scores
#     cv_results.append([
#         name,
#         np.mean(r2_scores),
#         np.std(r2_scores),
#         np.mean(mae_scores),
#         np.mean(mse_scores),
#         np.mean(rmse_scores)
#     ])
#
# cv_df = pd.DataFrame(cv_results,
#                      columns=["Model", "Mean CV R²", "Std Dev CV R²", "Mean CV MAE", "Mean CV MSE", "Mean CV RMSE"])
# st.dataframe(cv_df.set_index("Model"))
#
# # --- VISUALIZATION OF CROSS-VALIDATION RESULTS ---
# st.subheader("R² Scores for Each Cross-Validation Fold")
# fig_cv = go.Figure()
# for name, scores in r2_scores_dict.items():
#     fig_cv.add_trace(go.Box(y=scores, name=name))
#
# fig_cv.update_layout(title="R² Scores Distribution across 5 Folds",
#                      yaxis_title="R² Score",
#                      showlegend=False)
# st.plotly_chart(fig_cv, use_container_width=True)
#
# # Train final models and plot Actual vs Predicted
# st.subheader("Actual vs Predicted Scores on Test Set")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Fit", line=dict(color="white", dash='dot')))
#
# predictions = {}
# for name, model in models.items():
#     if name == "Simple Linear Regression":
#         X_train_specific = X_train[["Hours_Studied"]]
#         X_test_specific = X_test[["Hours_Studied"]]
#     else:
#         X_train_specific = X_train
#         X_test_specific = X_test
#
#     model.fit(X_train_specific, y_train)
#     y_pred = model.predict(X_test_specific)
#     predictions[name] = y_pred
#     fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name=name))
#
# fig.update_layout(title="Actual vs Predicted Scores", xaxis_title="Actual", yaxis_title="Predicted")
# st.plotly_chart(fig, use_container_width=True)
#
# # Calculate and show test set performance metrics
# st.subheader("Test Set Performance Metrics")
#
# test_metrics = []
# for name, y_pred in predictions.items():
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#
#     test_metrics.append([name, r2, mae, mse, rmse])
#
# test_metrics_df = pd.DataFrame(test_metrics,
#                                columns=["Model", "R²", "MAE", "MSE", "RMSE"])
# st.dataframe(test_metrics_df.set_index("Model"))
