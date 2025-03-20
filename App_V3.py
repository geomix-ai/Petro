import streamlit as st
from streamlit_file_browser import st_file_browser
import io
import os
import lasio
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib
from sklearn.tree import plot_tree

# Streamlit page configuration
st.set_page_config(page_title="🤖 Petrophysics Expert Robot", layout="wide")

# Sidebar Menu
st.sidebar.title("🤖 Petrophysics Robot")
if st.sidebar.button("🏠 Home / Introduction"):
    st.session_state.menu_choice = "Home"
if st.sidebar.button("⬇️ Load File"):
    st.session_state.menu_choice = "Load File"
if st.sidebar.button("📺 Show Input Logs"):
    st.session_state.menu_choice = "Show Input Logs"
if st.sidebar.button("🛠️ Fix Logs"):
    st.session_state.menu_choice = "Fix Logs"
if st.sidebar.button("🎯 Select Training Data"):
    st.session_state.menu_choice = "Select Training Data"
if st.sidebar.button("📊 Plot Histograms"):
    st.session_state.menu_choice = "Plot Histograms"
if st.sidebar.button("📈 Plot Correlation Matrix"):
    st.session_state.menu_choice = "Plot Correlation Matrix"
if st.sidebar.button("</> Train Models & Show Predictions"):
    st.session_state.menu_choice = "Train Models and Show Predictions"
if st.sidebar.button("⌛ Load & Predict New Data"):
    st.session_state.menu_choice = "Load & Predict New Data"

# Initialize session state for menu selection
if "menu_choice" not in st.session_state:
    st.session_state.menu_choice = "Home"  # Default to Home page

# Initialize session state for global variables
if "df" not in st.session_state:
    st.session_state["df"] = None
if "target_log" not in st.session_state:
    st.session_state["target_log"] = None
if "input_logs" not in st.session_state:
    st.session_state["input_logs"] = None
if "models" not in st.session_state:
    st.session_state["models"] = {
        "Linear Regression": None,
        "Random Forest": None,
        "Neural Network": None,
        "XGBoost": None,
        "SVR": None,
        "KNN": None
    }
if "updated_X" not in st.session_state:
    st.session_state["updated_X"] = None
if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None

# Home / Introduction Page
if st.session_state.menu_choice == "Home":
    st.title("Welcome to 🤖 Petrophysics Expert Robot")
    st.write("""
    This application is designed to assist petrophysicists in analyzing log data, training machine learning models,
    and making predictions. Follow these steps to use the application effectively:

    1. **Load Data:** Upload LAS or CSV files containing well log data using the `Load File` button.
    2. **View & Clean Logs:** Fix missing values and outliers through `Show Input Logs` and `Fix Logs`.
    3. **Select Training Data:** Choose the target log and input logs for model training.
    4. **Visualize & Analyze Data:** using histograms and correlation matrices.
    5. **Train Machine Learning Models:** Select ML models, tune hyperparameters, train and evaluate their performance.
    6. **Use Trained Models & Predict New Data:** Load new well logs and make predictions using the trained models.

    Navigate through the menu on the left to access different functionalities.
    """)

# Load LAS or CSV files
def load_file():
    uploaded_file = st.file_uploader("Upload LAS or CSV file", type=["las", "csv"])

    if not uploaded_file:
        st.warning("No file uploaded yet!")
        return

    try:
        if uploaded_file.size > 200 * 1024 * 1024:  # 200 MB limit
            st.error(f"File {uploaded_file.name} is too large! Max size is 200 MB.")
            return

        if uploaded_file.name.endswith(".las"):
            las = lasio.read(io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore")))
            temp_df = las.df()
        elif uploaded_file.name.endswith(".csv"):
            temp_df = pd.read_csv(uploaded_file, index_col=0)
        else:
            st.error(f"Unsupported file format: {uploaded_file.name}")
            return

        st.session_state["df"] = temp_df
        st.success(f"Loaded: {uploaded_file.name} ({len(temp_df)} rows)")

    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")

# Show input logs with interactive and colorful plots
def show_input_logs():
    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.subheader("Well Logs")

        # Create subplots with separate tracks for each log
        fig = make_subplots(
            rows=1,
            cols=len(df.columns),
            shared_yaxes=True,
            horizontal_spacing=0.02,
            subplot_titles=df.columns
        )

        for j, col in enumerate(df.columns):
            min_val = df[col].min()

            # Fill from log's minimum value to the log value
            fig.add_trace(
                go.Scatter(
                    x=[min_val] * len(df.index),
                    y=df.index,
                    mode='lines',
                    line=dict(color='red', width=0),  # Transparent line
                    showlegend=False,
                ),
                row=1,
                col=j + 1
            )

            fig.add_trace(
                go.Scatter(
                    x=df[col],
                    y=df.index,
                    mode='lines',
                    name=col,
                    line=dict(color='red', width=1),
                    fill='tonextx',
                    fillcolor='rgba(128, 128, 128, 0.3)',  # Semi-transparent grey fill
                ),
                row=1,
                col=j + 1
            )

            # Update each x-axis with fine grid
            fig.update_xaxes(
                title_text=col,
                row=1,
                col=j + 1,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='gray'
            )

        # General layout updates
        fig.update_yaxes(
            title="Depth (m)",
            autorange="reversed",
            showgrid=True,
            gridwidth=0.5,
            gridcolor='gray'
        )

        fig.update_layout(
            height=1000,  # Increase height for deep wells
            width=300 * len(df.columns),  # Adjust width dynamically
            title="Well Log Visualization",
            template="plotly_white",
            hovermode="y unified"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No data loaded!")

# Fix missing values
def fix_logs():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠ No data loaded!")
        return

    missing_values = st.text_input("Enter missing values to replace (comma separated, e.g., -999.25, -999)", "-999.25,-999,-9999")
    missing_values = [float(val.strip()) for val in missing_values.split(",")]

    df = st.session_state["df"]
    df.replace(missing_values, np.nan, inplace=True)
    fill_method = st.selectbox("Choose method to fill missing values", ["Drop Rows", "Fill with Mean", "Fill with Median", "Interpolate"])

    if st.button("Preview Changes"):
        st.write("Before Cleaning:")
        st.write(df.head())

    if fill_method == "Drop Rows":
        df.dropna(inplace=True)
    elif fill_method == "Fill with Mean":
        df.fillna(df.mean(), inplace=True)
    elif fill_method == "Fill with Median":
        df.fillna(df.median(), inplace=True)
    elif fill_method == "Interpolate":
        df.interpolate(inplace=True)

    st.session_state["cleaned_df"] = df
    st.success("✔ Data cleaned successfully!")
    show_input_logs()

    # Save Cleaned Logs with Depth Interval Selection
    save_option = st.radio("Save Option", ["Entire Logs", "Specific Depth Interval"])

    if save_option == "Specific Depth Interval":
        depth_min = st.number_input("Enter Minimum Depth", value=float(df.index.min()))
        depth_max = st.number_input("Enter Maximum Depth", value=float(df.index.max()))
        if depth_min >= depth_max:
            st.error("Minimum depth must be less than maximum depth!")
        else:
            if st.button("Save Cleaned Logs"):
                cleaned_df = df[(df.index >= depth_min) & (df.index <= depth_max)]
                st.session_state["cleaned_df"] = cleaned_df
                st.success(f"✔ Cleaned logs saved for depth interval {depth_min} to {depth_max}!")
    else:
        if st.button("Save Cleaned Logs"):
            st.session_state["cleaned_df"] = df
            st.success("✔ Cleaned logs saved for the entire depth range!")

# Select target and input logs for Training
def select_training_data():
    if "cleaned_df" not in st.session_state or st.session_state["cleaned_df"] is None:
        st.warning("⚠ No cleaned data available!")
        return

    st.write("### Select Training Data")
    df = st.session_state["cleaned_df"]

    st.session_state["target_log"] = st.selectbox("Select Target Log:", df.columns)
    st.session_state["input_logs"] = st.multiselect("Select Input Logs:", df.columns,
                                                     default=[col for col in df.columns if col != st.session_state["target_log"]])

    if st.button("Confirm Selection"):
        if not st.session_state["target_log"] or not st.session_state["input_logs"]:
            st.warning("⚠ Please select both input and target logs!")
        else:
            st.success(f"✔ Logs selected successfully!\nTarget: {st.session_state['target_log']}\nInputs: {st.session_state['input_logs']}")

# Plot histograms of input logs and target log
def plot_histograms():
    if "input_logs" not in st.session_state or "target_log" not in st.session_state:
        st.warning("⚠ No training data selected!")
        return

    input_logs = st.session_state["input_logs"]
    target_log = st.session_state["target_log"]

    if "cleaned_df" not in st.session_state or st.session_state["cleaned_df"] is None:
        st.warning("⚠ No cleaned data available!")
        return

    if input_logs and target_log:
        st.write("### Histograms")
        df = st.session_state["cleaned_df"]
        fig, axes = plt.subplots(nrows=1, ncols=len(input_logs) + 1, figsize=(25, 5))

        for i, col in enumerate(input_logs):
            if col in df.columns:
                axes[i].hist(df[col].dropna(), bins=30, edgecolor='black')
                axes[i].set_title(col)

        if target_log in df.columns:
            axes[-1].hist(df[target_log].dropna(), bins=30, edgecolor='black', color='red')
            axes[-1].set_title(target_log)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("⚠ No data loaded or logs selected!")

# Plot correlation matrix and update X data
def plot_correlation_matrix():
    if "cleaned_df" not in st.session_state or st.session_state["cleaned_df"] is None:
        st.warning("⚠ No cleaned data available!")
        return

    if "input_logs" not in st.session_state or not st.session_state["input_logs"]:
        st.warning("⚠ No logs selected!")
        return

    input_logs = st.session_state["input_logs"]
    df = st.session_state["cleaned_df"]

    if input_logs:
        st.write("### Correlation Matrix and Selected Input Logs")
        corr_matrix = df[input_logs].corr()

        high_corr = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr.add(corr_matrix.columns[i])

        st.session_state["updated_X"] = df[input_logs].drop(columns=high_corr)

        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, ax=ax_corr, cmap="coolwarm")
        ax_corr.set_title("Correlation Matrix")
        st.pyplot(fig_corr)

        fig, axes = plt.subplots(nrows=1, ncols=len(st.session_state["updated_X"].columns), figsize=(10, 10))
        for i, col in enumerate(st.session_state["updated_X"].columns):
            axes[i].plot(st.session_state["updated_X"][col], st.session_state["updated_X"].index, label=col)
            axes[i].set_ylim(st.session_state["updated_X"].index.max(), st.session_state["updated_X"].index.min())
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Depth")
            axes[i].grid()
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.warning("⚠ No logs selected!")

# Train Models and Show Predictions
def train_models_and_show_predictions():
    if "cleaned_df" not in st.session_state or st.session_state["cleaned_df"] is None:
        st.warning("⚠ No cleaned data available!")
        return

    if "input_logs" not in st.session_state or "target_log" not in st.session_state:
        st.warning("⚠ No logs selected!")
        return

    input_logs = st.session_state["input_logs"]
    target_log = st.session_state["target_log"]

    if st.session_state["cleaned_df"] is not None and input_logs and target_log:
        model_name = st.selectbox("Choose Model", list(st.session_state["models"].keys()))

        # Set Hyperparameters
        param_grid = {}
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 1, 20, 10)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            param_grid = {"n_estimators": range(10, 200, 10), "max_depth": range(1, 20)}
        elif model_name == "Neural Network":
            hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 64,64)", "64,64")
            max_iter = st.slider("Max Iterations", 100, 1000, 100)
            model = MLPRegressor(hidden_layer_sizes=tuple(map(int, hidden_layer_sizes.split(','))), max_iter=max_iter, random_state=42)
            param_grid = {"hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 128)], "max_iter": range(100, 1000, 100)}
        elif model_name == "XGBoost":
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 1, 20, 6)
            model = xgb.XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            param_grid = {
                "learning_rate": np.linspace(0.01, 0.3, 10),
                "n_estimators": range(10, 200, 10),
                "max_depth": range(1, 20)
            }
        elif model_name == "SVR":
            kernel = st.text_input("Kernel (e.g., 'rbf', 'linear')", "rbf")
            C = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0)
            gamma = st.text_input("Gamma (Kernel coefficient)", "scale")
            model = SVR(kernel=kernel, C=C, gamma=gamma)
            param_grid = {"C": np.linspace(0.1, 10, 10), "gamma": ["scale", "auto"]}
        elif model_name == "KNN":
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            param_grid = {"n_neighbors": range(1, 20)}

        use_random_search = st.checkbox("Use RandomizedSearchCV for Hyperparameter Tuning")

        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                df = st.session_state["cleaned_df"]
                X = st.session_state["updated_X"].dropna() if st.session_state["updated_X"] is not None else df[input_logs].dropna()
                y = df[target_log].dropna()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                X_scaled = scaler.transform(X)

                if use_random_search and param_grid:
                    search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, random_state=42)
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    st.success(f"Best hyperparameters: {search.best_params_}")
                else:
                    model.fit(X_train, y_train)

                st.session_state["models"][model_name] = model
                st.success(f"{model_name} trained successfully!")

                # Show Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                y_pred = model.predict(X_scaled)

                # Calculate Metrics
                metrics_data = {
                    "Dataset": ["Training", "Testing"],
                    "R²": [r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)],
                    "RMSE": [np.sqrt(mean_squared_error(y_train, y_pred_train)),
                             np.sqrt(mean_squared_error(y_test, y_pred_test))]
                }
                metrics_df = pd.DataFrame(metrics_data)

                # Plot Predictions
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y.index,
                    y=y.values,
                    mode='lines',
                    name='Actual',
                    line=dict(color='rgba(0, 0, 0, 0.7)', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=y.index,
                    y=y_pred,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='rgba(255, 0, 0, 0.7)', width=2)
                ))
                # Update layout
                fig.update_layout(
                    title=f"{model_name} (R²: {metrics_data['R²'][1]:.2f}, RMSE: {metrics_data['RMSE'][1]:.2f})",
                    xaxis_title="Depth",
                    yaxis_title="Values",
                    legend_title="Legend",
                    hovermode="x unified",
                    template="plotly_white"
                )

                # Display the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)

                # Show Metrics Table
                col1, col2 = st.columns(2)
                col1.metric("R² (Training)", f"{metrics_data['R²'][0]:.2f}")
                col2.metric("RMSE (Training)", f"{metrics_data['RMSE'][0]:.2f}")

                col3, col4 = st.columns(2)
                col3.metric("R² (Testing)", f"{metrics_data['R²'][1]:.2f}")
                col4.metric("RMSE (Testing)", f"{metrics_data['RMSE'][1]:.2f}")

                # Save Model
                if st.button("Save Model"):
                    try:
                        model_path = f"{model_name}_model.pkl"
                        with open(model_path, "wb") as file:
                            joblib.dump(model, file)
                        st.session_state["model_saved"] = True
                        st.session_state["model_path"] = model_path
                    except Exception as e:
                        st.error(f"Error saving model: {e}")

                # Display success message if model is saved
                if st.session_state.get("model_saved"):
                    st.success(f"Model saved successfully at: {st.session_state['model_path']}")
    else:
        st.warning("⚠ No data or logs selected!")

# Load and predict new data
def load_and_predict_new_data():
    uploaded_file = st.file_uploader("Upload new LAS or CSV file", type=["las", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith(".las"):
            las = lasio.read(io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore")))
            new_df = las.df()
        elif uploaded_file.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded_file)

        if "Depth" in new_df.columns:
            new_df.set_index("Depth", inplace=True)

        input_logs_new = st.multiselect("Select Input Logs for Prediction", new_df.columns)
        if input_logs_new:
            X_new = new_df[input_logs_new].dropna()
            X_new_scaled = StandardScaler().fit_transform(X_new)

            predictions = {}
            for model_name, model in st.session_state["models"].items():
                if model is not None:
                    y_pred_new = model.predict(X_new_scaled)
                    predictions[model_name] = y_pred_new

            pred_df = pd.DataFrame(predictions, index=X_new.index)
            pred_df["Depth"] = pred_df.index

            st.write("New Logs")
            fig, axes = plt.subplots(nrows=1, ncols=len(input_logs_new), figsize=(15, 6))
            for i, col in enumerate(input_logs_new):
                axes[i].plot(new_df[col], new_df.index, label=col)
                axes[i].set_ylim(new_df.index.max(), new_df.index.min())
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Depth")
                axes[i].grid()
            plt.tight_layout()
            st.pyplot(fig)

            st.write("Predicted Log")
            fig = go.Figure()
            for model_name in pred_df.columns:
                if model_name != "Depth":
                    fig.add_trace(go.Scatter(
                        x=pred_df["Depth"],
                        y=pred_df[model_name],
                        mode='lines',
                        name=model_name
                    ))
            fig.update_layout(
                xaxis_title="Depth",
                yaxis_title="Predicted Values",
                title="Predicted Log",
                legend_title="Model",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Export Results using streamlit-file-browser
            if st.button("Export Results"):
                st.write("Select a folder and enter a file name to save the results:")

                # Set the initial path for the file browser
                initial_path = os.getcwd()  # Use the current working directory as the initial path
                selected_path = st_file_browser(path=initial_path)

                if selected_path:
                    file_name = st.text_input("Enter file name (e.g., Results.las or Results.csv)")
                    if file_name:
                        if not file_name.endswith((".las", ".csv")):
                            st.error("Invalid file format! Use .las or .csv.")
                        else:
                            export_path = os.path.join(selected_path, file_name)
                            try:
                                if file_name.endswith(".las"):
                                    las = lasio.LASFile()
                                    las.set_data_from_df(pred_df)
                                    las.write(export_path)
                                elif file_name.endswith(".csv"):
                                    pred_df.to_csv(export_path, index=False)
                                st.success(f"Results exported successfully to {export_path}!")
                            except Exception as e:
                                st.error(f"Error exporting results: {e}")
    else:
        st.warning("No file selected!")
        
# Main UI
st.title("💡 Petrophysical Property Predictor")

# Execute the selected function
if st.session_state.menu_choice == "Load File":
    load_file()
elif st.session_state.menu_choice == "Show Input Logs":
    show_input_logs()
elif st.session_state.menu_choice == "Fix Logs":
    fix_logs()
elif st.session_state.menu_choice == "Select Training Data":
    select_training_data()
elif st.session_state.menu_choice == "Plot Histograms":
    plot_histograms()
elif st.session_state.menu_choice == "Plot Correlation Matrix":
    plot_correlation_matrix()
elif st.session_state.menu_choice == "Train Models and Show Predictions":
    train_models_and_show_predictions()
elif st.session_state.menu_choice == "Load & Predict New Data":
    load_and_predict_new_data()
