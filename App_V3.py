import streamlit as st
import io
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
import pickle

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
if "dfs" not in st.session_state:
    st.session_state["dfs"] = []
if "cleaned_dfs" not in st.session_state:
    st.session_state["cleaned_dfs"] = []
if "target_logs" not in st.session_state:
    st.session_state["target_logs"] = []
if "input_logs" not in st.session_state:
    st.session_state["input_logs"] = []
if "models" not in st.session_state:
    st.session_state["models"] = {
        "Linear Regression": None,
        "Random Forest": None,
        "Neural Network": None,
        "XGBoost": None,
        "SVR": None,
        "KNN": None
    }
if "average_metrics" not in st.session_state:
    st.session_state["average_metrics"] = {}

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
    uploaded_files = st.file_uploader("Upload LAS or CSV files", type=["las", "csv"], accept_multiple_files=True)

    if not uploaded_files:
        st.warning("No file uploaded yet!")
        return

    st.session_state["dfs"] = []
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.size > 200 * 1024 * 1024:  # 200 MB limit
                st.error(f"File {uploaded_file.name} is too large! Max size is 200 MB.")
                continue

            if uploaded_file.name.endswith(".las"):
                las = lasio.read(io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore")))
                temp_df = las.df()
            elif uploaded_file.name.endswith(".csv"):
                temp_df = pd.read_csv(uploaded_file, index_col=0)
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                continue

            st.session_state["dfs"].append(temp_df)
            st.success(f"Loaded: {uploaded_file.name} ({len(temp_df)} rows)")

        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")

# Show input logs with interactive and colorful plots
def show_input_logs():
    if "dfs" in st.session_state and st.session_state["dfs"]:
        for i, df in enumerate(st.session_state["dfs"]):
            st.subheader(f"Well {i+1} Logs")

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
                        x=[min_val]*len(df.index),
                        y=df.index,
                        mode='lines',
                        line=dict(color='red', width=0),  # Transparent line
                        showlegend=False,
                    ),
                    row=1,
                    col=j+1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df[col],
                        y=df.index,
                        mode='lines',
                        name=col,
                        line=dict(color='black', width=1),
                        fill='tonextx',
                        fillcolor='rgba(128, 128, 128, 0.3)',  # Semi-transparent grey fill
                    ),
                    row=1,
                    col=j+1
                )

                # Update each x-axis with fine grid
                fig.update_xaxes(
                    title_text=col,
                    row=1,
                    col=j+1,
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
                title=f"Well {i+1} - Log Visualization",
                template="plotly_white",
                hovermode="y unified"
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No data loaded!")

# Fix missing values
def fix_logs():
    if "dfs" not in st.session_state or not st.session_state["dfs"]:
        st.warning("⚠ No data loaded!")
        return

    missing_values = st.text_input("Enter missing values to replace (comma separated, e.g., -999.25, -999)", "-999.25,-999,-9999")
    missing_values = [float(val.strip()) for val in missing_values.split(",")]

    cleaned_dfs = []
    for i, df in enumerate(st.session_state["dfs"]):
        df.replace(missing_values, np.nan, inplace=True)
        fill_method = st.selectbox(
            f"Choose method to fill missing values for Well {i+1}", 
            ["Drop Rows", "Fill with Mean", "Fill with Median", "Interpolate"],
            key=f"fill_method_selectbox_{i}"  # Unique key
        )

        if st.button(f"Preview Changes for Well {i+1}", key=f"preview_button_{i}"):  # Unique key
            st.write(f"Before Cleaning for Well {i+1}:")
            st.write(df.head())

        if fill_method == "Drop Rows":
            df.dropna(inplace=True)
        elif fill_method == "Fill with Mean":
            df.fillna(df.mean(), inplace=True)
        elif fill_method == "Fill with Median":
            df.fillna(df.median(), inplace=True)
        elif fill_method == "Interpolate":
            df.interpolate(inplace=True)

        cleaned_dfs.append(df)

    st.session_state["cleaned_dfs"] = cleaned_dfs
    st.success("✔ Data cleaned successfully!")
    show_input_logs()

    if st.button("Save Cleaned Logs"):
        for i, df in enumerate(cleaned_dfs):
            df.to_csv(f"cleaned_well_{i+1}.csv", index=False)
        st.success("✔ Cleaned logs saved for each well!")
    else:
        st.warning("⚠ Cleaned logs not saved!")

# Select target and input logs for Training
def select_training_data():
    if "cleaned_dfs" not in st.session_state or not st.session_state["cleaned_dfs"]:
        st.warning("⚠ No cleaned data available!")
        return

    st.write("### Select Training Data for Each Well")
    st.session_state["target_logs"] = []
    st.session_state["input_logs"] = []

    for i, df in enumerate(st.session_state["cleaned_dfs"]):
        st.subheader(f"Well {i+1}")
        target_log = st.selectbox(
            f"Select Target Log for Well {i+1}:", 
            df.columns,
            key=f"target_log_selectbox_{i}"  # Unique key
        )
        input_logs = st.multiselect(
            f"Select Input Logs for Well {i+1}:", 
            df.columns,
            default=[col for col in df.columns if col != target_log],
            key=f"input_logs_multiselect_{i}"  # Unique key
        )

        st.session_state["target_logs"].append(target_log)
        st.session_state["input_logs"].append(input_logs)

    if st.button("Confirm Selection"):
        st.success("✔ Logs selected successfully for all wells!")

# Plot histograms of input logs and target log for each well
def plot_histograms():
    if "cleaned_dfs" not in st.session_state or not st.session_state["cleaned_dfs"]:
        st.warning("⚠ No cleaned data available!")
        return

    if "target_logs" not in st.session_state or "input_logs" not in st.session_state:
        st.warning("⚠ No logs selected!")
        return

    for i, df in enumerate(st.session_state["cleaned_dfs"]):
        st.subheader(f"Well {i+1} Histograms")
        target_log = st.session_state["target_logs"][i]
        input_logs = st.session_state["input_logs"][i]

        fig, axes = plt.subplots(nrows=1, ncols=len(input_logs) + 1, figsize=(25, 5))

        for j, col in enumerate(input_logs):
            if col in df.columns:
                axes[j].hist(df[col].dropna(), bins=30, edgecolor='black')
                axes[j].set_title(col)

        if target_log in df.columns:
            axes[-1].hist(df[target_log].dropna(), bins=30, edgecolor='black', color='red')
            axes[-1].set_title(target_log)

        plt.tight_layout()
        st.pyplot(fig)

# Plot correlation matrix and selected input logs for each well
def plot_correlation_matrix():
    if "cleaned_dfs" not in st.session_state or not st.session_state["cleaned_dfs"]:
        st.warning("⚠ No cleaned data available!")
        return

    if "input_logs" not in st.session_state or not st.session_state["input_logs"]:
        st.warning("⚠ No logs selected!")
        return

    for i, df in enumerate(st.session_state["cleaned_dfs"]):
        st.subheader(f"Well {i+1} Correlation Matrix and Selected Input Logs")
        input_logs = st.session_state["input_logs"][i]

        if input_logs:
            corr_matrix = df[input_logs].corr()

            high_corr = set()
            for j in range(len(corr_matrix.columns)):
                for k in range(j):
                    if abs(corr_matrix.iloc[j, k]) > 0.8:
                        high_corr.add(corr_matrix.columns[j])

            st.session_state["updated_X"] = df[input_logs].drop(columns=high_corr)

            fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr_matrix, annot=True, ax=ax_corr, cmap="coolwarm")
            ax_corr.set_title("Correlation Matrix")
            st.pyplot(fig_corr)

            fig, axes = plt.subplots(nrows=1, ncols=len(st.session_state["updated_X"].columns), figsize=(10, 10))
            for j, col in enumerate(st.session_state["updated_X"].columns):
                axes[j].plot(st.session_state["updated_X"][col], st.session_state["updated_X"].index, label=col)
                axes[j].set_ylim(st.session_state["updated_X"].index.max(), st.session_state["updated_X"].index.min())
                axes[j].set_xlabel(col)
                axes[j].set_ylabel("Depth")
                axes[j].grid()
            plt.tight_layout()
            st.pyplot(fig)

# Train Models and Show Predictions for each well
def train_models_and_show_predictions():
    if "cleaned_dfs" not in st.session_state or not st.session_state["cleaned_dfs"]:
        st.warning("⚠ No cleaned data available!")
        return

    if "input_logs" not in st.session_state or "target_logs" not in st.session_state:
        st.warning("⚠ No logs selected!")
        return

    model_name = st.selectbox(
        "Choose Model", 
        list(st.session_state["models"].keys()),
        key="model_selectbox"  # Unique key
    )

    # Set Hyperparameters
    param_grid = {}
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100, key="rf_n_estimators_slider")
        max_depth = st.slider("Max Depth", 1, 20, 10, key="rf_max_depth_slider")
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        param_grid = {"n_estimators": range(10, 200, 10), "max_depth": range(1, 20)}
    elif model_name == "Neural Network":
        hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 64,64)", "64,64", key="nn_hidden_layer_sizes_input")
        max_iter = st.slider("Max Iterations", 100, 1000, 100, key="nn_max_iter_slider")
        model = MLPRegressor(hidden_layer_sizes=tuple(map(int, hidden_layer_sizes.split(','))), max_iter=max_iter, random_state=42)
        param_grid = {"hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 128)], "max_iter": range(100, 1000, 100)}
    elif model_name == "XGBoost":
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, key="xgb_learning_rate_slider")
        n_estimators = st.slider("Number of Trees", 10, 200, 100, key="xgb_n_estimators_slider")
        max_depth = st.slider("Max Depth", 1, 20, 6, key="xgb_max_depth_slider")
        model = xgb.XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        param_grid = {
            "learning_rate": np.linspace(0.01, 0.3, 10),
            "n_estimators": range(10, 200, 10),
            "max_depth": range(1, 20)
        }
    elif model_name == "SVR":
        kernel = st.text_input("Kernel (e.g., 'rbf', 'linear')", "rbf", key="svr_kernel_input")
        C = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0, key="svr_c_slider")
        gamma = st.text_input("Gamma (Kernel coefficient)", "scale", key="svr_gamma_input")
        model = SVR(kernel=kernel, C=C, gamma=gamma)
        param_grid = {"C": np.linspace(0.1, 10, 10), "gamma": ["scale", "auto"]}
    elif model_name == "KNN":
        n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, key="knn_n_neighbors_slider")
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        param_grid = {"n_neighbors": range(1, 20)}

    use_random_search = st.checkbox("Use RandomizedSearchCV for Hyperparameter Tuning", key="use_random_search_checkbox")

    if st.button("Train Model", key="train_model_button"):
        with st.spinner("Training in progress..."):
            metrics_data = {"Well": [], "R²": [], "RMSE": []}
            for i, df in enumerate(st.session_state["cleaned_dfs"]):
                target_log = st.session_state["target_logs"][i]
                input_logs = st.session_state["input_logs"][i]

                X = df[input_logs].dropna()
                y = df[target_log].dropna()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                if use_random_search and param_grid:
                    search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, random_state=42)
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    st.success(f"Best hyperparameters for Well {i+1}: {search.best_params_}")
                else:
                    model.fit(X_train, y_train)

                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                metrics_data["Well"].append(f"Well {i+1}")
                metrics_data["R²"].append(r2_score(y_test, y_pred_test))
                metrics_data["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred_test)))

            # Calculate average metrics
            avg_r2 = np.mean(metrics_data["R²"])
            avg_rmse = np.mean(metrics_data["RMSE"])
            st.session_state["average_metrics"][model_name] = {"R²": avg_r2, "RMSE": avg_rmse}

            st.success(f"{model_name} trained successfully for all wells!")
            st.write("### Average Metrics")
            st.write(f"Average R²: {avg_r2:.2f}")
            st.write(f"Average RMSE: {avg_rmse:.2f}")

# Load and predict new data using average values
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
            for model_name, metrics in st.session_state["average_metrics"].items():
                if model_name in st.session_state["models"] and st.session_state["models"][model_name] is not None:
                    y_pred_new = st.session_state["models"][model_name].predict(X_new_scaled)
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
            fig, ax = plt.subplots(figsize=(20, 5))
            for model_name in pred_df.columns:
                if model_name != "Depth":
                    ax.plot(pred_df["Depth"], pred_df[model_name], label=model_name)
            ax.set_xlabel("Depth")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Predicted Log")
            ax.legend()
            ax.grid()
            st.pyplot(fig)

            if st.button("Export Results"):
                export_path = st.text_input("Enter file path to save results (e.g., results.las or results.csv)")
                if export_path:
                    if not export_path.endswith((".las", ".csv")):
                        st.error("Invalid file format! Use .las or .csv.")
                    else:
                        if export_path.endswith(".las"):
                            las = lasio.LASFile()
                            las.set_data_from_df(pred_df)
                            las.write(export_path)
                        elif export_path.endswith(".csv"):
                            pred_df.to_csv(export_path, index=False)
                        st.success("Results exported successfully!")
    else:
        st.warning("No file selected!")

# Main UI
st.title("🧪 Petrophysical Property Predictor")

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
