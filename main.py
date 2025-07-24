import io
import os
import requests
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
import plotly.express as px
from dotenv import load_dotenv      
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load environment variables
load_dotenv()

# Set page Config
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CSV Analyzer", layout="wide")

# 🧠 Title & Intro
st.title("📊 Web CSV Data Analyzer")
st.markdown("*Comprehensive data analysis tool with advanced features*")

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
phase = st.sidebar.selectbox("Select Analysis Phase", [
    "Data Collection", 
    "Data Cleaning", 
    "Exploratory Data Analysis", 
    "Advanced Visualization", 
    "Statistical Analysis",
    "Data Export & Reports"
])

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Analysis Phases")
st.sidebar.markdown("""
1. **Data Collection**
2. **Data Cleaning** 
3. **Exploratory Data Analysis** 
4. **Advanced Visualization**
5. **Statistical Analysis** 
6. **Data Export & Reports** 
""")

#  Session state to store datasets and analysis results
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "final_df" not in st.session_state:
    st.session_state.final_df = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

shape=(0,0)
#  Phase 1: Data Collection
if phase == "Data Collection":
    st.header("Phase 1: Data Collection 📁")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load data with error handling
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.raw_df = df
            st.success(f"✅ Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Quick overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Dataset preview
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(10))
            shape = df.shape
            # Column information
            with st.expander("📊 Detailed Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2),
                    'Unique Values': df.nunique()
                })
                st.dataframe(col_info)
            
            # Basic statistics
            with st.expander("📈 Summary Statistics"):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.write("**Numeric Columns:**")
                    st.dataframe(df[numeric_cols].describe())
                
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    st.write("**Categorical Columns:**")
                    for col in categorical_cols[:5]:  # Show first 5 categorical columns
                        st.write(f"**{col}:** {df[col].nunique()} unique values")
                        st.write(df[col].value_counts().head())
                        st.write("---")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

#  Phase 2: Data Cleaning
elif phase == "Data Cleaning":
    st.header("Phase 2: Data Cleaning 🧹")
    
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first in the Data Collection phase.")
    else:
        # Initialize working dataframe
        if "working_df" not in st.session_state:
            st.session_state.working_df = st.session_state.raw_df.copy()
        
        df = st.session_state.working_df
        
        # Display current dataset info
        st.subheader("Current Dataset Status")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Reset button
        if st.button("🔄 Reset to Original Dataset"):
            st.session_state.working_df = st.session_state.raw_df.copy()
            st.success("Dataset reset to original state!")
            st.rerun()
        
        # Missing values handling
        st.subheader("🚫 Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Missing Values by Column:**")
                missing_df = pd.DataFrame({
                    'Missing Count': missing_data,
                    'Missing Percentage': (missing_data / len(df) * 100).round(2)
                })
                st.dataframe(missing_df)
            
            with col2:
                # Missing values visualization
                fig, ax = plt.subplots(figsize=(8, 6))
                missing_data.plot(kind='bar', ax=ax)
                ax.set_title('Missing Values by Column')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.clf()
        else:
            st.success("✅ No missing values found!")
        
        # Cleaning options
        st.subheader("🛠️ Basic Cleaning Operations")
        
        col1, col2 = st.columns(2)

        # Drop rows with missing values
        with col1:
            if st.button("Drop All Rows with Missing Values"):
                initial_rows = df.shape[0]
                st.session_state.working_df = df.dropna()
                removed_rows = initial_rows - st.session_state.working_df.shape[0]
                st.success(f"Removed {removed_rows} rows with missing values. New shape: {st.session_state.working_df.shape}")
                st.rerun()
        
        # Drop duplicate rows
        with col2:
            if st.button("Drop Duplicate Rows"):
                initial_rows = df.shape[0]
                st.session_state.working_df = df.drop_duplicates()
                removed_rows = initial_rows - st.session_state.working_df.shape[0]
                st.success(f"Removed {removed_rows} duplicate rows. New shape: {st.session_state.working_df.shape}")
                st.rerun()
        
        # Advanced missing value handling
        with st.expander("🎯 Advanced Missing Value Handling"):
            if len(missing_data) > 0:
                col_to_fill = st.selectbox("Select column to fill missing values:", missing_data.index)
                fill_method = st.selectbox("Fill method:", ["Forward Fill", "Backward Fill", "Mean", "Median", "Mode", "Custom Value"])
                
                if fill_method == "Custom Value":
                    custom_val = st.text_input("Enter custom value:")
                if st.button("Apply Fill Method"):
                    working_df = df.copy()
                    
                    # Fill methods 
                    try:
                        if fill_method == "Forward Fill":
                            working_df[col_to_fill] = working_df[col_to_fill].fillna(method='ffill')
                        elif fill_method == "Backward Fill":
                            working_df[col_to_fill] = working_df[col_to_fill].fillna(method='bfill')
                        elif fill_method == "Mean" and working_df[col_to_fill].dtype in ['int64', 'float64']:
                            working_df[col_to_fill] = working_df[col_to_fill].fillna(working_df[col_to_fill].mean())
                        elif fill_method == "Median" and working_df[col_to_fill].dtype in ['int64', 'float64']:
                            working_df[col_to_fill] = working_df[col_to_fill].fillna(working_df[col_to_fill].median())
                        elif fill_method == "Mode":
                            mode_value = working_df[col_to_fill].mode()
                            if len(mode_value) > 0:
                                working_df[col_to_fill] = working_df[col_to_fill].fillna(mode_value[0])
                        elif fill_method == "Custom Value":
                            if custom_val:
                                # Try to convert custom value to appropriate type
                                if working_df[col_to_fill].dtype in ['int64', 'float64']:
                                    try:
                                        custom_val = float(custom_val)
                                    except ValueError:
                                        st.error("Please enter a numeric value for numeric columns")
                                working_df[col_to_fill] = working_df[col_to_fill].fillna(custom_val)
                            else:
                                st.error("Please enter a custom value")
                        
                        st.session_state.working_df = working_df
                        filled_count = df[col_to_fill].isnull().sum() - working_df[col_to_fill].isnull().sum()
                        st.success(f"Applied {fill_method} to {col_to_fill}. Filled {filled_count} missing values.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying fill method: {str(e)}")
        
        # Data type conversion
        with st.expander("🔄 Data Type Conversion"):
            col_to_convert = st.selectbox("Select column to convert:", df.columns)
            current_type = str(df[col_to_convert].dtype)
            st.write(f"Current type: **{current_type}**")
            
            new_type = st.selectbox("New data type:", ["int64", "int32", "int16", "float64", "object", "datetime64", "category"])
            
            if st.button("Convert Data Type"):
                try:
                    working_df = df.copy()
                    if new_type == "datetime64":
                        working_df[col_to_convert] = pd.to_datetime(working_df[col_to_convert], errors='coerce')
                    else:
                        working_df[col_to_convert] = working_df[col_to_convert].astype(new_type)
                    
                    st.session_state.working_df = working_df
                    st.success(f"Converted {col_to_convert} from {current_type} to {new_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error converting data type: {str(e)}")
        
        # Column operations
        with st.expander("📝 Column Operations"):
            # Drop columns
            st.write("**Drop Columns:**")
            cols_to_drop = st.multiselect("Select columns to drop:", df.columns.tolist())
            if cols_to_drop and st.button("Drop Selected Columns"):
                working_df = df.drop(columns=cols_to_drop)
                st.session_state.working_df = working_df
                st.success(f"Dropped {len(cols_to_drop)} columns. New shape: {working_df.shape}")
                st.rerun()
            
            # Rename columns
            st.write("**Rename Column:**")
            col1, col2 = st.columns(2)
            with col1:
                old_name = st.selectbox("Column to rename:", df.columns.tolist(), key="rename_old")
            with col2:
                new_name = st.text_input("New name:", key="rename_new")
            
            if new_name and st.button("Rename Column"):
                if new_name not in df.columns:
                    working_df = df.rename(columns={old_name: new_name})
                    st.session_state.working_df = working_df
                    st.success(f"Renamed '{old_name}' to '{new_name}'")
                    st.rerun()
                else:
                    st.error(f"Column '{new_name}' already exists!")
        
        # Data filtering
        with st.expander("🔍 Data Filtering"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                filter_col = st.selectbox("Select column to filter:", numeric_cols)
                col_min = float(df[filter_col].min())
                col_max = float(df[filter_col].max())
                
                range_values = st.slider(
                    f"Select range for {filter_col}",
                    min_value=col_min,
                    max_value=col_max,
                    value=(col_min, col_max)
                )
                
                if st.button("Apply Filter"):
                    working_df = df[(df[filter_col] >= range_values[0]) & (df[filter_col] <= range_values[1])]
                    st.session_state.working_df = working_df
                    filtered_count = len(df) - len(working_df)
                    st.success(f"Filtered out {filtered_count} rows. New shape: {working_df.shape}")
                    st.rerun()
        
        # Preview current dataset
        st.subheader("Current Dataset Preview")
        st.write(f"**Current shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Show before/after comparison if changes were made
        if df.shape != st.session_state.raw_df.shape:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original:**")
                st.write(f"Shape: {st.session_state.raw_df.shape}")
                st.write(f"Missing values: {st.session_state.raw_df.isnull().sum().sum()}")
            with col2:
                st.write("**Current:**")
                st.write(f"Shape: {df.shape}")
                st.write(f"Missing values: {df.isnull().sum().sum()}")
        
        # Display sample of current data
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show data types
        with st.expander("📋 Current Data Types"):
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(dtype_df)
        
        # Final save button
        st.subheader("💾 Save Cleaned Dataset")
        if st.button("✅ Save as Final Dataset", type="primary"):
            st.session_state.final_df = df.copy()
            st.success(f"✅ Dataset saved as final! Shape: {df.shape}")
            st.success("You can now proceed to the analysis phases.")
            
            # Clear working dataframe to save memory
            if "working_df" in st.session_state:
                del st.session_state.working_df

#  Phase 3: Exploratory Data Analysis
elif phase == "Exploratory Data Analysis":
    st.header("Phase 3: Exploratory Data Analysis 🔍")
    
    if st.session_state.final_df is None:
        st.warning("Please complete data cleaning and save a final dataset first.")
    else:
        df = st.session_state.final_df
        
        # Data overview
        st.subheader("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", df.shape[0])
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Categorical Features", len(df.select_dtypes(include=['object']).columns))
        
        # Distribution analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("📈 Distribution Analysis")
            selected_col = st.selectbox("Select numeric column for distribution analysis:", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(df[selected_col], kde=True, ax=ax)
                ax.set_title(f'Distribution of {selected_col}')
                st.pyplot(fig)
                plt.clf()
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(y=df[selected_col], ax=ax)
                ax.set_title(f'Box Plot of {selected_col}')
                st.pyplot(fig)
                plt.clf()
            
            # Statistical summary
            col_stats = df[selected_col].describe()
            st.write("**Statistical Summary:**")
            st.dataframe(col_stats.to_frame().T)
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            st.subheader("📊 Categorical Analysis")
            selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
            
            value_counts = df[selected_cat_col].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Value Counts:**")
                st.dataframe(value_counts.to_frame())
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                value_counts.head(10).plot(kind='bar', ax=ax)
                ax.set_title(f'Top 10 Values in {selected_cat_col}')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.clf()

#  Phase 4: Advanced Visualization
elif phase == "Advanced Visualization":
    st.header("Phase 4: Advanced Visualization 📊")
    
    if st.session_state.final_df is None:
        st.warning("Please complete data cleaning and save a final dataset first.")
    else:
        df = st.session_state.final_df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Correlation Analysis
        if len(numeric_cols) > 1:
            st.subheader("🔗 Correlation Analysis")
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)
            plt.clf()
        
        # Interactive Scatter Plot
        if len(numeric_cols) >= 2:
            st.subheader("🎯 Interactive Scatter Plot")
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis:", numeric_cols)
            with col2:
                y_col = st.selectbox("Y-axis:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            with col3:
                color_col = st.selectbox("Color by:", ["None"] + categorical_cols)
            
            if color_col != "None":
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f'{y_col} vs {x_col}')
            else:
                fig = px.scatter(df, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Time Series Analysis (if datetime columns exist)
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            st.subheader("📅 Time Series Analysis")
            date_col = st.selectbox("Select date column:", datetime_cols)
            value_col = st.selectbox("Select value column:", numeric_cols)
            
            fig = px.line(df, x=date_col, y=value_col, title=f'{value_col} over Time')
            st.plotly_chart(fig, use_container_width=True)
        
        # Advanced Plots
        st.subheader("📈 Advanced Visualizations")
        plot_type = st.selectbox("Select plot type:", [
            "Pair Plot", "Distribution Grid", "Box Plot Comparison", "Violin Plot"
        ])
        
        if plot_type == "Pair Plot" and len(numeric_cols) >= 2:
            selected_numeric = st.multiselect("Select numeric columns:", numeric_cols, default=numeric_cols[:4])
            if selected_numeric:
                fig = sns.pairplot(df[selected_numeric])
                st.pyplot(fig)
                plt.clf()
        
        # Box Plot Comparison
        elif plot_type == "Box Plot Comparison" and categorical_cols and numeric_cols:
            cat_col = st.selectbox("Categorical column:", categorical_cols)
            num_col = st.selectbox("Numeric column:", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.clf()

# #  Phase 5: Statistical Analysis
elif phase == "Statistical Analysis":
    st.header("Phase 5: Statistical Analysis 📊")
    
    if st.session_state.final_df is None:
        st.warning("Please complete data cleaning and save a final dataset first.")
    else:
        df = st.session_state.final_df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Correlation Analysis
            st.subheader("🔗 Detailed Correlation Analysis")
            corr_method = st.selectbox("Correlation method:", ["pearson", "spearman", "kendall"])
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr(method=corr_method)
            
            # Display correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title(f'{corr_method.title()} Correlation Matrix')
            st.pyplot(fig)
            plt.clf()
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            st.write("**Strongest Correlations:**")
            st.dataframe(corr_df.head(10))
            
            # Hypothesis Testing
            st.subheader("🧪 Hypothesis Testing")
            if len(numeric_cols) >= 2:
                col1_test = st.selectbox("First variable:", numeric_cols, key="test1")
                col2_test = st.selectbox("Second variable:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="test2")
                
                if st.button("Perform Correlation Test"):
                    # Remove NaN values for testing
                    test_data1 = df[col1_test].dropna()
                    test_data2 = df[col2_test].dropna()
                    
                    # Align the data
                    common_idx = test_data1.index.intersection(test_data2.index)
                    test_data1 = test_data1.loc[common_idx]
                    test_data2 = test_data2.loc[common_idx]
                    
                    if len(test_data1) > 3:
                        try:
                            corr_coef, p_value = stats.pearsonr(test_data1, test_data2)
                            
                            st.write(f"**Pearson Correlation Coefficient:** {corr_coef:.4f}")
                            st.write(f"**P-value:** {p_value:.4f}")
                            
                            if p_value < 0.05:
                                st.success("✅ Significant correlation detected (p < 0.05)")
                            else:
                                st.info("❌ No significant correlation (p >= 0.05)")
                        except Exception as e:
                            st.error(f"Error in correlation test: {str(e)}")
                    else:
                        st.warning("Not enough data points for correlation test (need > 3)")
            
            # Simple Linear Regression
            st.subheader("📈 Simple Linear Regression")
            if len(numeric_cols) >= 2:
                target_col = st.selectbox("Target variable (Y):", numeric_cols, key="target")
                feature_col = st.selectbox("Feature variable (X):", [col for col in numeric_cols if col != target_col], key="feature")
                
                if st.button("Perform Regression Analysis"):
                    # Prepare data
                    reg_data = df[[feature_col, target_col]].dropna()
                    
                    if len(reg_data) > 10:
                        try:
                            X = reg_data[[feature_col]]
                            y = reg_data[target_col]
                            
                            # Split data for validation
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Fit model
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            y_pred_train = model.predict(X_train)
                            y_pred_test = model.predict(X_test)
                            
                            # Calculate metrics
                            r2_train = r2_score(y_train, y_pred_train)
                            r2_test = r2_score(y_test, y_pred_test)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Training R²:** {r2_train:.4f}")
                                st.write(f"**Test R²:** {r2_test:.4f}")
                                st.write(f"**Coefficient:** {model.coef_[0]:.4f}")
                                st.write(f"**Intercept:** {model.intercept_:.4f}")
                            
                            with col2:
                                # Plot regression line
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.scatter(X, y, alpha=0.6, label='Data points')
                                ax.plot(X, model.predict(X), color='red', linewidth=2, label='Regression line')
                                ax.set_xlabel(feature_col)
                                ax.set_ylabel(target_col)
                                ax.set_title(f'Linear Regression: {target_col} vs {feature_col}')
                                ax.legend()
                                st.pyplot(fig)
                                plt.clf()
                        
                        except Exception as e:
                            st.error(f"Error in regression analysis: {str(e)}")
                    else:
                        st.warning("Not enough data points for regression analysis (need > 10)")
        
        st.subheader('🤖 Machine Learning Analysis')
        
        # Target variable selection
        target_column = st.selectbox("Select the target column", options=df.columns, key="ml_target")
        
        if target_column:
            # Determine problem type
            target_unique = df[target_column].nunique()
            is_numeric_target = df[target_column].dtype in ['int64', 'float64']
            
            if is_numeric_target and target_unique > 10:
                problem_type = "Regression"
                st.info("🎯 **Problem Type:** Regression (continuous target)")
            else:
                problem_type = "Classification" 
                st.info(f"🎯 **Problem Type:** Classification ({target_unique} classes)")
            
            # Feature preparation
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Show data info
            st.write(f"**Features:** {X.shape[1]} columns, **Target:** {target_column}")
            st.write(f"**Dataset size:** {len(df)} rows")
            
            # Handle missing values and data types
            with st.expander("🔧 Data Preprocessing Options"):
                st.write("**Categorical Columns:**", categorical_cols if categorical_cols else "None")
                st.write("**Numeric Columns:**", numeric_cols if numeric_cols else "None")
                
                # Encoding option for categorical variables
                if any(col in X.columns for col in categorical_cols):
                    encode_categorical = st.checkbox("Encode categorical variables", value=True)
                else:
                    encode_categorical = False
                
                # Handle missing values option
                handle_missing = st.selectbox("Handle missing values:", ["Drop rows", "Fill with mean/mode"])
            
            try:
                X_processed = X.copy()
                
                # Handle missing values
                if handle_missing == "Drop rows":
                    initial_size = len(X_processed)
                    X_processed = X_processed.dropna()
                    y = y.loc[X_processed.index]
                    st.write(f"Dropped {initial_size - len(X_processed)} rows with missing values")
                else:
                    # Fill numeric columns with mean, categorical with mode
                    for col in X_processed.columns:
                        if X_processed[col].dtype in ['int64', 'float64']:
                            X_processed[col].fillna(X_processed[col].mean(), inplace=True)
                        else:
                            mode_val = X_processed[col].mode()
                            if len(mode_val) > 0:
                                X_processed[col].fillna(mode_val[0], inplace=True)
                
                # Encode categorical variables
                if encode_categorical:
                    categorical_in_X = [col for col in categorical_cols if col in X_processed.columns]
                    if categorical_in_X:
                        X_processed = pd.get_dummies(X_processed, columns=categorical_in_X, drop_first=True)
                        st.write(f"Encoded {len(categorical_in_X)} categorical columns")
                
                # Check if we have valid data
                if len(X_processed) == 0:
                    st.error("No data remaining after preprocessing!")
                elif X_processed.shape[1] == 0:
                    st.error("No features remaining after preprocessing!")
                else:
                    # Train-test split
                    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y, test_size=test_size, random_state=42, stratify=y if problem_type == "Classification" else None
                    )
                    
                    # Model selection based on problem type
                    if problem_type == "Classification":
                        model_options = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM']
                        default_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                    else:
                        model_options = ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'SVR']
                        default_metrics = ['R²', 'MAE', 'MSE', 'RMSE']
                    
                    model_type = st.selectbox('Select Model Type', model_options)
                    
                    if st.button("🚀 Train Model"):
                        try:
                            # Model initialization
                            if model_type == "Logistic Regression":
                                from sklearn.linear_model import LogisticRegression
                                model = LogisticRegression(max_iter=1000, random_state=42)
                            elif model_type == "Decision Tree":
                                from sklearn.tree import DecisionTreeClassifier
                                model = DecisionTreeClassifier(random_state=42)
                            elif model_type == "Random Forest":
                                from sklearn.ensemble import RandomForestClassifier
                                model = RandomForestClassifier(n_estimators=100, random_state=42)
                            elif model_type == "SVM":
                                from sklearn.svm import SVC
                                model = SVC(random_state=42)
                            elif model_type == "Linear Regression":
                                from sklearn.linear_model import LinearRegression
                                model = LinearRegression()
                            elif model_type == "Decision Tree Regressor":
                                from sklearn.tree import DecisionTreeRegressor
                                model = DecisionTreeRegressor(random_state=42)
                            elif model_type == "Random Forest Regressor":
                                from sklearn.ensemble import RandomForestRegressor
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                            elif model_type == "SVR":
                                from sklearn.svm import SVR
                                model = SVR()
                            
                            # Train model
                            model.fit(X_train, y_train)
                            
                            # Make predictions
                            y_pred_train = model.predict(X_train)
                            y_pred_test = model.predict(X_test)
                            
                            st.success('🎉 Model trained successfully!')
                            
                            # Calculate and display metrics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("📊 Training Metrics")
                                if problem_type == "Classification":
                                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                                    train_acc = accuracy_score(y_train, y_pred_train)
                                    st.metric("Accuracy", f"{train_acc:.4f}")
                                    
                                    if target_unique == 2:
                                        # Binary classification
                                        train_prec = precision_score(y_train, y_pred_train, average='binary')
                                        train_rec = recall_score(y_train, y_pred_train, average='binary')
                                        train_f1 = f1_score(y_train, y_pred_train, average='binary')
                                    else:
                                        # Multi-class
                                        train_prec = precision_score(y_train, y_pred_train, average='weighted')
                                        train_rec = recall_score(y_train, y_pred_train, average='weighted')
                                        train_f1 = f1_score(y_train, y_pred_train, average='weighted')
                                    
                                    st.metric("Precision", f"{train_prec:.4f}")
                                    st.metric("Recall", f"{train_rec:.4f}")
                                    st.metric("F1-Score", f"{train_f1:.4f}")
                                
                                else:
                                    # Regression
                                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                                    train_r2 = r2_score(y_train, y_pred_train)
                                    train_mae = mean_absolute_error(y_train, y_pred_train)
                                    train_mse = mean_squared_error(y_train, y_pred_train)
                                    train_rmse = np.sqrt(train_mse)
                                    
                                    st.metric("R²", f"{train_r2:.4f}")
                                    st.metric("MAE", f"{train_mae:.4f}")
                                    st.metric("MSE", f"{train_mse:.4f}")
                                    st.metric("RMSE", f"{train_rmse:.4f}")
                            
                            with col2:
                                # Classification
                                st.subheader("🎯 Test Metrics")
                                if problem_type == "Classification":
                                    test_acc = accuracy_score(y_test, y_pred_test)
                                    st.metric("Accuracy", f"{test_acc:.4f}")
                                    
                                    if target_unique == 2:
                                        test_prec = precision_score(y_test, y_pred_test, average='binary')
                                        test_rec = recall_score(y_test, y_pred_test, average='binary')
                                        test_f1 = f1_score(y_test, y_pred_test, average='binary')
                                    else:
                                        test_prec = precision_score(y_test, y_pred_test, average='weighted')
                                        test_rec = recall_score(y_test, y_pred_test, average='weighted')
                                        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
                                    
                                    st.metric("Precision", f"{test_prec:.4f}")
                                    st.metric("Recall", f"{test_rec:.4f}")
                                    st.metric("F1-Score", f"{test_f1:.4f}")
                                
                                else: 
                                    # Regression
                                    test_r2 = r2_score(y_test, y_pred_test)
                                    test_mae = mean_absolute_error(y_test, y_pred_test)
                                    test_mse = mean_squared_error(y_test, y_pred_test)
                                    test_rmse = np.sqrt(test_mse)
                                    
                                    st.metric("R²", f"{test_r2:.4f}")
                                    st.metric("MAE", f"{test_mae:.4f}")
                                    st.metric("MSE", f"{test_mse:.4f}")
                                    st.metric("RMSE", f"{test_rmse:.4f}")
                            
                            # Additional visualizations
                            if problem_type == "Classification":
                                # Confusion Matrix
                                from sklearn.metrics import confusion_matrix
                                cm = confusion_matrix(y_test, y_pred_test)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                ax.set_title('Confusion Matrix')
                                ax.set_xlabel('Predicted')
                                ax.set_ylabel('Actual')
                                st.pyplot(fig)
                                plt.clf()
                            
                            else: 
                                # Regression
                                # Actual vs Predicted plot
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.scatter(y_test, y_pred_test, alpha=0.6)
                                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                                ax.set_xlabel('Actual')
                                ax.set_ylabel('Predicted')
                                ax.set_title('Actual vs Predicted Values')
                                st.pyplot(fig)
                                plt.clf()
                            
                            # Feature importance (for tree-based models)
                            if hasattr(model, 'feature_importances_'):
                                st.subheader("🎯 Feature Importance")
                                feature_importance = pd.DataFrame({
                                    'Feature': X_processed.columns,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.barplot(data=feature_importance.head(10), y='Feature', x='Importance', ax=ax)
                                ax.set_title('Top 10 Feature Importances')
                                st.pyplot(fig)
                                plt.clf()
                                
                                st.dataframe(feature_importance.head(10))
                        
                        except Exception as e:
                            st.error(f"Error during model training: {str(e)}")
                            st.write("**Possible issues:**")
                            st.write("- Target variable may not be suitable for selected model type")
                            st.write("- Data may contain incompatible values")
                            st.write("- Try different preprocessing options")
                            
            except Exception as e:
                st.error(f"Error during preprocessing: {str(e)}")
        
        # Statistical Tests Section
        if len(numeric_cols) >= 1:
            st.subheader("📈 Additional Statistical Tests")
            
            with st.expander("🔍 Normality Tests"):
                test_col = st.selectbox("Select column for normality test:", numeric_cols, key="normality_col")
                if st.button("Perform Shapiro-Wilk Test"):
                    test_data = df[test_col].dropna()
                    if len(test_data) > 3 and len(test_data) <= 5000:
                        try:
                            statistic, p_value = stats.shapiro(test_data)
                            st.write(f"**Shapiro-Wilk Statistic:** {statistic:.4f}")
                            st.write(f"**P-value:** {p_value:.4f}")
                            
                            if p_value > 0.05:
                                st.success("✅ Data appears to be normally distributed (p > 0.05)")
                            else:
                                st.warning("❌ Data does not appear to be normally distributed (p ≤ 0.05)")
                        except Exception as e:
                            st.error(f"Error in normality test: {str(e)}")
                    else:
                        st.warning("Sample size should be between 3 and 5000 for Shapiro-Wilk test")
            
            if len(numeric_cols) >= 2 and categorical_cols:
                with st.expander("📊 Group Comparisons"):
                    group_col = st.selectbox("Grouping variable:", categorical_cols, key="group_col")
                    value_col = st.selectbox("Value variable:", numeric_cols, key="value_col")
                    
                    if st.button("Perform ANOVA"):
                        try:
                            groups = df.groupby(group_col)[value_col].apply(list)
                            if len(groups) >= 2:
                                f_stat, p_value = stats.f_oneway(*groups.values)
                                st.write(f"**F-statistic:** {f_stat:.4f}")
                                st.write(f"**P-value:** {p_value:.4f}")
                                
                                if p_value < 0.05:
                                    st.success("✅ Significant difference between groups (p < 0.05)")
                                else:
                                    st.info("❌ No significant difference between groups (p ≥ 0.05)")
                            else:
                                st.warning("Need at least 2 groups for ANOVA")
                        except Exception as e:
                            st.error(f"Error in ANOVA: {str(e)}")
        
        # Save analysis results
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = {}
        
        # Store results for reporting
        st.session_state.analysis_results["statistical_analysis"] = {
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "correlations_calculated": len(numeric_cols) >= 2,
            "ml_analysis_available": True
        }
            

#  Phase 6: Data Export & Reports
elif phase == "Data Export & Reports":
    st.header("Phase 6: Data Export & Reports 💾")

    if st.session_state.final_df is None:
        st.warning("Please complete data cleaning and save a final dataset first.")
    else:
        df = st.session_state.final_df
        shape = df.shape

        # Export options
        st.subheader("📤 Export Data")

        col1, col2 = st.columns(2)

        with col1:
            # CSV export
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv_buffer.getvalue(),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

        # Report generation block
        st.subheader("📋 Analysis Report")

        if st.button("Generate Analysis Report"):
            report = f"""
            # Data Analysis Report

            ## Dataset Overview
            - **Total Records:** {df.shape[0]:,}
            - **Total Features:** {df.shape[1]}
            - **Missing Values:** {df.isnull().sum().sum()}

            ## Data Types
            {df.dtypes.value_counts().to_string()}

            ## Numeric Summary
            {df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else "No numeric columns"}

            ## Missing Values Summary
            {df.isnull().sum().to_string()}

            ## Data Quality Assessment
            - **Completeness:** {((df.count().sum() / (df.shape[0] * df.shape[1])) * 100):.2f}%
            - **Duplicate Rows:** {df.duplicated().sum()}
            """
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")

            client = genai.Client(api_key = api_key)

            try:
                def response():
                    response = client.models.generate_content(
                        model=os.getenv("MODEL_NAME"),
                        contents=f"Summarize the key insights from the analysis report below. Focus on highlighting major findings, trends, and any actionable recommendations. Provide clarity on the implications of these insights. Analysis report: {report}"
                    )
                    return response
                insights = response()

            except requests.exceptions.RequestException as e:
                insights = f"❌ Error communicating with the API: {e}"

            st.markdown("### 🔍 Key Insights")
            st.markdown(insights.text)

            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name="analysis_report.txt",
                mime="text/plain"
            )

        st.sidebar.info(f"Shape of dataset: {shape[0]} rows × {shape[1]} columns")