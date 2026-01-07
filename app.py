import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import lightgbm as lgb
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import clone


st.set_page_config(layout="wide")

st.title("🏭 Air Quality Analytics & Long-Range Risk Outlook")
st.write("""
This dashboard evaluates historical model performance and provides a 1-year risk outlook to identify future periods with historically high air pollution.
Select a station from the sidebar for location-specific analysis.
""")

@st.cache_data
def load_and_prep_data():
    all_files = glob.glob(os.path.join('C:/Users/mhdda/air-quality-project', "*.csv"))
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    station_names = df['station'].unique().tolist()
    df.dropna(inplace=True)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df.drop(['year', 'month', 'day', 'hour', 'No'], axis=1, inplace=True)
    return df, station_names

@st.cache_resource
def train_models(df):
    df_featured = df.copy()

    # Time-based features
    df_featured['hour'] = df_featured.index.hour
    df_featured['day_of_week'] = df_featured.index.dayofweek
    
    # Lag features
    features_to_lag = ['PM2.5', 'TEMP', 'WSPM']
    lag_periods = [1, 24, 48]
    for feature in features_to_lag:
        for lag in lag_periods:
            df_featured[f'{feature}_lag_{lag}'] = df_featured[feature].shift(lag)
            
    # Rolling statistics
    df_featured['PM2.5_roll_mean_24h'] = df_featured['PM2.5'].rolling(window=24, min_periods=1).mean()
    df_featured['PM2.5_roll_std_24h'] = df_featured['PM2.5'].rolling(window=24, min_periods=1).std()
    
    # Advanced interaction features
    df_featured['stagnant_air_index'] = df_featured['PRES'] / (df_featured['WSPM'] + 1)
    df_featured['is_rush_hour'] = df_featured['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df_featured['PM2.5_to_PM10_ratio'] = df_featured['PM2.5'] / (df_featured['PM10'] + 0.001)
    df_featured['NO2_to_CO_ratio'] = df_featured['NO2'] / (df_featured['CO'] + 0.001)
    
    df_featured.dropna(inplace=True)

    # One-hot encode the categorical features
    df_encoded = pd.get_dummies(df_featured, columns=['wd', 'station'], drop_first=True)

    targets = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    y = df_encoded[targets]
    X = df_encoded.drop(columns=targets)
    split_point = int(len(X) * 0.9)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    alpha_values = {
        'PM2.5': 0.88, 'PM10': 0.85, 'SO2': 0.45,
        'NO2': 0.85, 'CO': 0.8, 'O3': 0.75
    }
    models_to_train = {
        "LightGBM_Quantile": lgb.LGBMRegressor(objective='quantile', random_state=42, n_jobs=-1),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10),
        "LinearRegression": LinearRegression(n_jobs=-1) 
    }
    trained_models = {}
    for model_name, model_template in models_to_train.items(): 
        st.write(f"Training {model_name}...")
        models_for_pollutants = {}
        for target in targets:
            current_model = clone(model_template)
            
            if model_name == "LightGBM_Quantile":
                current_model.set_params(alpha=alpha_values.get(target, 0.8))
            
            current_model.fit(X_train, y_train[target])
           
            models_for_pollutants[target] = current_model
            
        trained_models[model_name] = models_for_pollutants
            
    return trained_models, X_test, y_test, targets

# --- Main App Logic ---
df, station_names = load_and_prep_data()
models, X_test, y_test, targets = train_models(df)

st.sidebar.header("📍 Filter Options")
selected_station_name = st.sidebar.selectbox("Select a station to view:", station_names)

station_column_name = f"station_{selected_station_name}"
if station_column_name in X_test.columns:
    station_mask = (X_test[station_column_name] == 1)
else:
    station_cols = [col for col in X_test.columns if col.startswith('station_')]
    station_mask = (X_test[station_cols].sum(axis=1) == 0)
X_test_filtered = X_test[station_mask]
y_test_filtered = y_test[station_mask]

tab1, tab2, tab3 = st.tabs(["🗓️ 1-Year Risk Outlook", "⚙️ Historical Model Performance", "🧠 Model Interpretation"])

with tab1:
    st.header(f"1-Year Multi-Pollutant Risk Outlook for {selected_station_name}")
    st.write("This tool identifies weeks in the year following the last data record with a high historical probability of unhealthy air quality for PM2.5, NO2, or SO2.")

    station_data = df[df['station'] == selected_station_name]
    numeric_cols = station_data.select_dtypes(include=np.number).columns
    daily_data = station_data[numeric_cols].resample('D').mean().dropna()

    unhealthy_thresholds = {'PM2.5': 150.4, 'NO2': 100, 'SO2': 80}

    daily_data_copy = daily_data.copy()
    for pollutant, threshold in unhealthy_thresholds.items():
        daily_data_copy[f'{pollutant}_unhealthy'] = daily_data_copy[pollutant] > threshold
    
    unhealthy_cols = [f'{p}_unhealthy' for p in unhealthy_thresholds.keys()]
    daily_data_copy['is_unhealthy_day'] = daily_data_copy[unhealthy_cols].any(axis=1)
    daily_data_copy['week_num'] = daily_data_copy.index.isocalendar().week
    full_year_weekly_risk = daily_data_copy.groupby('week_num')['is_unhealthy_day'].mean() * 100
    
    def get_high_pollutants(week_df):
        unhealthy_days = week_df[week_df['is_unhealthy_day']]
        if unhealthy_days.empty: return ""
        high_pollutant_flags = unhealthy_days[unhealthy_cols].any()
        return ', '.join(high_pollutant_flags.index[high_pollutant_flags].str.replace('_unhealthy', ''))

    high_pollutant_summary = daily_data_copy.groupby('week_num').apply(get_high_pollutants)

    last_record_date = df.index.max()
    st.info(f"Outlook based on data ending {last_record_date.strftime('%Y-%m-%d')}.")
    
    current_week = last_record_date.isocalendar().week
    upcoming_weeks_indices = [((current_week + i - 1) % 52) + 1 for i in range(52)]
    outlook_risk_for_next_year = full_year_weekly_risk.reindex(upcoming_weeks_indices)
    high_risk_weeks = outlook_risk_for_next_year[outlook_risk_for_next_year > 25]

    st.subheader("High-Risk Weeks Identified for the Next Year")
    if not high_risk_weeks.empty:
        st.error(f"**Warning:** Based on historical patterns, the following weeks have a high risk of unhealthy air quality:")
        high_risk_weeks_df = high_risk_weeks.reset_index()
        high_risk_weeks_df.columns = ['Week Number', 'Probability of Unhealthy Days (%)']
        high_risk_weeks_df['High-Concentration Pollutants'] = high_risk_weeks_df['Week Number'].map(high_pollutant_summary)
        
        def get_month_for_week(week_num):
            month_map = {1: 'Jan', 5: 'Feb', 9: 'Mar', 14: 'Apr', 18: 'May', 22: 'Jun', 27: 'Jul', 31: 'Aug', 35: 'Sep', 40: 'Oct', 44: 'Nov', 48: 'Dec'}
            for start_week, month_name in reversed(list(month_map.items())):
                if week_num >= start_week:
                    return month_name
            return "Jan"
            
        high_risk_weeks_df['Likely Month'] = high_risk_weeks_df['Week Number'].apply(get_month_for_week)
        high_risk_weeks_df = high_risk_weeks_df[['Week Number', 'Likely Month', 'Probability of Unhealthy Days (%)', 'High-Concentration Pollutants']]
        st.dataframe(high_risk_weeks_df.style.format({'Probability of Unhealthy Days (%)': '{:.1f}%'}), hide_index=True)
    else:
        st.success("No weeks with a historically high risk of unhealthy air quality were identified in the next year.")


with tab2:
    st.header(f"📈 Model Performance Comparison at {selected_station_name}")
    if X_test_filtered.empty:
        st.warning("No test data available for this station in the historical test period.")
    else:
        all_metrics_data = []
        for model_name, trained_pollutant_models in models.items():
            for target in targets:
                predictions = trained_pollutant_models[target].predict(X_test_filtered)
                mae = mean_absolute_error(y_test_filtered[target], predictions)
                rmse = np.sqrt(mean_squared_error(y_test_filtered[target], predictions))
                value_range = y_test_filtered[target].max() - y_test_filtered[target].min()
                
                if value_range > 0:
                    nrmse = (rmse / value_range) * 100
                else:
                    nrmse = 0

                all_metrics_data.append({
                    'Model': model_name,
                    'Pollutant': target,
                    'MAE': mae,
                    'RMSE': rmse,
                    'NRMSE (%)': nrmse
                })
        
        metrics_df = pd.DataFrame(all_metrics_data)
        
        st.subheader("Performance Metrics Overview")
        st.info("ℹ️ **NRMSE** (Normalized Root Mean Squared Error) shows the error as a percentage of the data's range. Lower is better, and it helps compare performance across pollutants with different scales.")
        st.dataframe(metrics_df.style.format({
            'MAE': '{:.2f}', 
            'RMSE': '{:.2f}',
            'NRMSE (%)': '{:.2f}%'
        }))

        st.subheader("📊 Model Performance Bar Chart")

        metric_to_plot = 'NRMSE (%)'

        if metric_to_plot:
            fig_bar = px.bar(
                metrics_df,
                x='Pollutant',
                y=metric_to_plot,
                color='Model',
                barmode='group', 
                title=f'Comparison of {metric_to_plot} Across Models'
            )
            fig_bar.update_layout(yaxis_title=metric_to_plot)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        
        st.subheader(f"📊 Predictions vs. Actuals at {selected_station_name}")
        
        col1, col2 = st.columns(2)
        with col1:
            pollutant_to_plot = st.selectbox("Select a pollutant to visualize:", targets, key='perf_pollutant')
        with col2:
            model_to_plot = st.selectbox("Select a model to visualize:", list(models.keys()), key='perf_model')

        
        predictions_filtered = models[model_to_plot][pollutant_to_plot].predict(X_test_filtered)
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=y_test_filtered.index, y=y_test_filtered[pollutant_to_plot], mode='lines', name='Actual', line=dict(width=2)))
        fig_perf.add_trace(go.Scatter(x=y_test_filtered.index, y=predictions_filtered, mode='lines', name=f'{model_to_plot} Predicted', line=dict(dash='dot', width=2)))
        
        fig_perf.update_layout(title=f'"{model_to_plot}" Predictions for {pollutant_to_plot}', legend=dict(x=0, y=1.1))
        st.plotly_chart(fig_perf, use_container_width=True)

with tab3:
    st.header(f"🧠 Model Interpretation for {selected_station_name}")
    st.info("ℹ️ How to read this plot: Each dot is a prediction for a single hour. Red dots indicate a high value for that feature. Dots on theright pushed the prediction higher, while dots on the left pushed it lower.")
    
    if X_test_filtered.empty:
        st.warning("No data available for SHAP analysis for this station.")
    else:

        
        pollutant_to_explain = st.selectbox(
            "Select a pollutant model to interpret:",
            ('PM2.5', 'NO2'), 
            key='shap_pollutant_select'
        )

        st.write(f"This plot shows the most significant features influencing the **{pollutant_to_explain}** model's predictions.")

        with st.expander(f"Click here to generate SHAP plot for {pollutant_to_explain}"):
            with st.spinner(f"Calculating feature importance for {pollutant_to_explain}..."):
                
                selected_model = models['LightGBM_Quantile'][pollutant_to_explain]
                
                explainer = shap.TreeExplainer(selected_model)
                shap_values = explainer.shap_values(X_test_filtered)
                
                fig_shap, ax_shap = plt.subplots(tight_layout=True)
                
                shap.summary_plot(shap_values, X_test_filtered, show=False, plot_size=(6,4), max_display=10) 
                
                st.pyplot(fig_shap)
        
