# main.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import AI logic from a separate file
from ai_logic import generate_llm_insight # Ensure ai_logic.py is in the same directory

# --- Configuration ---
st.set_page_config(
    page_title="SLA Breach Predictor Dashboard",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling (DARK MODE & SLEEK DESIGN) ---
st.markdown("""
<style>
    /* Dark Mode & Sleek Design */
    body {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        color: #F0F0F0; /* Light text */
        background-color: #1E1E1E; /* Dark background */
    }

    .stApp {
        background-color: #1E1E1E;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #F0F0F0; /* Light text for headers */
    }

    /* Streamlit's default block padding for better use of space */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem; /* Larger font for tab names */
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #2C2C2C; /* Darker grey for unselected tabs */
        border-radius: 8px 8px 0 0; /* More rounded corners */
        margin: 0 5px;
        padding: 10px 20px;
        color: #BBBBBB; /* Lighter grey text for unselected tabs */
        border: 1px solid #3A3A3A; /* Subtle border */
        border-bottom: none; /* No bottom border for unselected */
        transition: all 0.2s ease-in-out; /* Smooth transition */
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #3A3A3A; /* Slightly lighter on hover */
        color: #F0F0F0;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1E1E1E; /* Main background for selected tab */
        color: #F0F0F0; /* White text for selected tab */
        border-bottom: 3px solid #4A90E2; /* Vibrant blue indicator for selected tab */
        border-top: 1px solid #4A90E2;
        border-left: 1px solid #4A90E2;
        border-right: 1px solid #4A90E2;
    }


    /* General Markdown styling */
    .stMarkdown p {
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* Metric Boxes */
    .metric-box {
        background-color: #2C2C2C; /* Darker background for metric cards */
        padding: 20px; /* More padding */
        border-radius: 12px; /* More rounded corners */
        box-shadow: 0 4px 8px rgba(0,0,0,0.3); /* Deeper shadow for dark mode depth */
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #3A3A3A; /* Subtle border */
    }
    .metric-title {
        font-size: 1.1rem;
        color: #AAAAAA; /* Muted grey text for titles */
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2.5rem; /* Larger font */
        font-weight: bold;
        color: #F0F0F0; /* Light text for values */
    }

    /* LLM Insight Box */
    .llm-insight-box {
        background-color: #2C2C2C; /* Match card background */
        border-left: 5px solid #4A90E2; /* Vibrant blue border */
        padding: 25px; /* More padding */
        border-radius: 12px; /* More rounded corners */
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin-top: 30px; /* More margin top */
        color: #F0F0F0;
        border: 1px solid #3A3A3A; /* Subtle border */
    }
    .llm-insight-box strong {
        color: #4A90E2; /* Vibrant blue for emphasis */
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #222222; /* Slightly different dark for sidebar */
        border-right: 1px solid #3A3A3A; /* Subtle separator */
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6, [data-testid="stSidebar"] p {
        color: #F0F0F0; /* Ensure text in sidebar is light */
    }

    /* Buttons */
    .stButton>button {
        background-color: #4A90E2; /* Vibrant blue */
        color: white;
        padding: 12px 25px; /* More padding */
        border: none;
        border-radius: 8px; /* More rounded */
        cursor: pointer;
        font-size: 1.1rem; /* Slightly larger font */
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease; /* Smooth transitions */
    }
    .stButton>button:hover {
        background-color: #3A7ADB; /* Darker blue on hover */
        transform: translateY(-2px); /* Slight lift effect */
    }
    .stButton>button:active {
        transform: translateY(0); /* Reset on click */
    }

    /* Sliders and other input fields */
    .stSlider > div > div > div > div {
        background-color: #4A90E2; /* Vibrant blue fill for slider */
    }
    .stSlider > div > div > div > div > div {
        background-color: #F0F0F0; /* Light handle for slider */
        border: 2px solid #4A90E2; /* Blue border for handle */
    }
    .stNumberInput > div > div > input, .stDateInput > div > div > input, .stFileUploader > section > div > p {
        background-color: #333333; /* Dark input background */
        color: #F0F0F0; /* Light text */
        border: 1px solid #555555; /* Subtle border */
        border-radius: 8px; /* More rounded */
        padding: 10px 15px; /* More padding */
    }
    .stFileUploader > section {
        background-color: #2C2C2C; /* Match card background */
        border: 2px dashed #4A90E2; /* Vibrant dashed border */
        border-radius: 12px; /* More rounded */
        padding: 25px; /* More padding */
    }
    .stFileUploader > section > div > p {
        color: #BBBBBB; /* Light grey text */
        font-size: 1.1rem;
    }
    /* Style for info/warning boxes */
    .stAlert {
        background-color: #2C2C2C !important;
        color: #F0F0F0 !important;
        border-left: 5px solid !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    .stAlert.info { border-color: #4A90E2 !important; }
    .stAlert.warning { border-color: orange !important; }
    .stAlert.success { border-color: #28A745 !important; }
    .stAlert.error { border-color: #E53935 !important; }
</style>
""", unsafe_allow_html=True)


# --- Data Generation Function (for sample data) ---
@st.cache_data
def generate_sample_data(num_records=1000):
    np.random.seed(42) # for reproducibility
    start_times = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365 * 24 * 60, num_records), unit='minutes')
    
    # Base duration of pipeline runs
    durations_base = np.random.normal(loc=40, scale=15, size=num_records)
    durations_base[durations_base < 5] = 5 # Minimum duration
    
    # Simulate some additional delay for a 'delay_minutes' column
    delay_minutes = np.random.normal(loc=5, scale=10, size=num_records)
    delay_minutes[delay_minutes < 0] = 0 # No negative delays

    # Introduce some seasonal delay (e.g., higher delays towards month-end or specific days)
    # Get day of month for each start_time
    day_of_month = pd.Series(start_times).dt.day
    # Apply a higher delay factor if it's nearing month-end (e.g., day 25-31)
    month_end_indices = day_of_month[day_of_month >= 25].index
    delay_minutes[month_end_indices] = delay_minutes[month_end_indices] * 1.5 + 10 # Add more delay for month-end

    # Total duration is base duration + simulated delay
    actual_duration_minutes = durations_base + delay_minutes
    
    end_times = start_times + pd.to_timedelta(actual_duration_minutes, unit='minutes')

    # Simulate success/failure based on actual duration and random chance
    # More likely to fail if duration is very high
    success_prob = 1 - (actual_duration_minutes / 120).clip(0, 0.5) + 0.5 # Higher success for shorter runs
    success = np.random.rand(num_records) < success_prob

    df = pd.DataFrame({
        'dag_id': np.random.choice(['pipeline_A', 'pipeline_B', 'pipeline_C', 'pipeline_D', 'pipeline_E', 'pipeline_F', 'pipeline_G'], num_records),
        'start_time': start_times,
        'end_time': end_times,
        'duration_minutes': actual_duration_minutes, # This is the actual total run time
        'success': success,
        'delay_minutes': delay_minutes # This represents the 'extra' delay beyond a theoretical ideal
    })

    return df

# --- Main Dashboard Layout ---
st.title("⏰ SLA Breach Predictor Dashboard")
st.markdown("This dashboard helps predict potential SLA breaches for data pipelines and provides AI-driven insights.")

# --- Data Upload/Selection Panel ---
st.sidebar.header("⚙️ Configuration")
with st.sidebar:
    st.subheader("⬆️ Data Input")
    uploaded_file = st.file_uploader(
        "Upload Historical Pipeline Run Data (CSV)",
        type=["csv"],
        help="Expected columns: `dag_id`, `start_time`, `end_time`, `success` (boolean), `duration_minutes` (actual run time), `delay_minutes` (actual delay)."
    )

    df = None # Initialize df to None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            df = generate_sample_data() # Fallback to sample data on error
    else:
        st.info("No file uploaded. Using a sample dataset for demonstration.")
        df = generate_sample_data()

    # --- Robust Column Handling ---
    if df is not None:
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()

        required_cols = ['start_time', 'end_time', 'dag_id', 'duration_minutes', 'success', 'delay_minutes']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.warning(f"The loaded data is missing the following recommended columns: **{', '.join(missing_cols)}**. For best results, please ensure your CSV has these columns or consider regenerating sample data.")
            # If start_time or end_time are missing, fallback to sample data completely.
            if 'start_time' not in df.columns or 'end_time' not in df.columns:
                st.error("Critical time columns (`start_time`, `end_time`) are missing after initial loading. Falling back to sample data for full functionality.")
                df = generate_sample_data()
                # Re-clean and re-check columns for sample data as a fail-safe
                df.columns = df.columns.str.strip()
                for col in ['start_time', 'end_time']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                df = df.dropna(subset=['start_time', 'end_time'])
            else:
                pass # Already warned above, proceed with current df if only non-critical are missing

        # Ensure datetime columns are correctly parsed for the final df
        for col in ['start_time', 'end_time']:
            if col in df.columns: # Check again after potential fallback
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                st.error(f"Column `'{col}'` is still missing after all data loading and cleaning attempts. Cannot proceed with time-based analysis. Please ensure your data has '{col}'.")
                st.stop() # Stop Streamlit execution if critical columns are truly absent

        # Drop rows where start_time or end_time could not be parsed (became NaT)
        df = df.dropna(subset=['start_time', 'end_time'])
        
        # Ensure 'duration_minutes' and 'delay_minutes' are numeric, handling NaNs if introduced by coerce
        for col in ['duration_minutes', 'delay_minutes']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Fill NaNs with 0 if they appear
            else:
                # If these are missing even after fallback to sample data, regenerate sample data entirely
                st.error(f"Required numeric column '{col}' is missing or invalid. Regenerating sample data.")
                df = generate_sample_data()
                df.columns = df.columns.str.strip() # Reclean after regeneration
                for dt_col in ['start_time', 'end_time']:
                    df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
                df = df.dropna(subset=['start_time', 'end_time'])
                for num_col in ['duration_minutes', 'delay_minutes']:
                    df[num_col] = pd.to_numeric(df[num_col], errors='coerce').fillna(0)
                break # Break and re-run through the checks for the newly generated df


    if df.empty:
        st.error("No valid data to display after loading and cleaning. Please upload a valid CSV or ensure sample data generation is working.")
        st.stop() # Stop if DataFrame is empty after processing


    st.subheader("🗓️ Time Range Selection")
    min_date_available = df['start_time'].min().date()
    max_date_available = df['start_time'].max().date()

    # Set default date range to something reasonable if data is very wide
    default_start_date = max_date_available - timedelta(days=30) if max_date_available - timedelta(days=30) >= min_date_available else min_date_available

    date_range = st.date_input(
        "Select Date Range for Analysis",
        value=(default_start_date, max_date_available), # Default to last 30 days or full range
        min_value=min_date_available,
        max_value=max_date_available
    )

    df_filtered = pd.DataFrame() # Initialize to empty
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['start_time'].dt.date >= start_date) & (df['start_time'].dt.date <= end_date)].copy()
    else:
        st.warning("Please select a start and end date for analysis.")
        df_filtered = df.copy() # Use full dataset if range isn't complete yet

    # Critical check: if df_filtered is empty after date selection
    if df_filtered.empty:
        st.warning("No data available for the selected date range. Please adjust your date selection or upload a different file.")
        st.stop() # Stop execution if no data is available after filtering

    sla_threshold = st.slider(
        "SLA Threshold (minutes)",
        min_value=10,
        max_value=120,
        value=60,
        step=5,
        help="Pipelines taking longer than this duration will be considered an SLA breach."
    )


# --- SLA Breach Prediction Logic (Placeholder - this is where an ML model would go) ---
# For now, a simple rule-based "prediction"
# A pipeline is "predicted" to breach if its last run or average run time is over threshold
# For a real project, this would involve training an ML model (e.g., XGBoost, Prophet, ARIMA)
# to forecast future run times or predict breach likelihood based on features.

@st.cache_data
def predict_breaches(data, threshold):
    # This is a simplified prediction:
    # Identify pipelines that have breached the SLA in the past based on actual duration.
    # We'll consider a "prediction" of future breach if their average duration *or* last run duration exceeds the threshold.

    # Calculate current breaches based on actual duration
    data['is_breach'] = data['duration_minutes'] > threshold

    # Get the latest run for each pipeline
    # Ensure 'start_time' is numeric for idxmax if not already (though should be datetime)
    latest_runs_indices = data.groupby('dag_id')['start_time'].idxmax()
    latest_runs = data.loc[latest_runs_indices].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Calculate average duration for each pipeline
    avg_durations = data.groupby('dag_id')['duration_minutes'].mean().reset_index()
    avg_durations.rename(columns={'duration_minutes': 'avg_duration_minutes'}, inplace=True)

    # Merge latest run info and average durations
    breach_pred_df = pd.merge(
        latest_runs[['dag_id', 'duration_minutes', 'success', 'delay_minutes']],
        avg_durations,
        on='dag_id',
        how='left'
    )
    breach_pred_df.rename(columns={'duration_minutes': 'last_run_duration_minutes'}, inplace=True)

    # A pipeline is "predicted to breach" if its last run or average duration exceeds the threshold
    breach_pred_df['predicted_to_breach'] = (breach_pred_df['last_run_duration_minutes'] > threshold) | \
                                            (breach_pred_df['avg_duration_minutes'] > threshold)

    breaching_pipelines_report = breach_pred_df[breach_pred_df['predicted_to_breach']].copy()

    # Calculate a simple 'likelihood_percentage' for the report
    if not breaching_pipelines_report.empty:
        # Calculate max_relevant_duration for each breaching pipeline (max of last_run or avg_duration)
        max_relevant_duration = np.maximum(
            breaching_pipelines_report['last_run_duration_minutes'],
            breaching_pipelines_report['avg_duration_minutes']
        )
        
        # Calculate how much it exceeds the threshold
        exceed_amount = max_relevant_duration - threshold
        
        # Normalize and scale to 50-100%
        # Divide by threshold to get a ratio of exceedance, then clip to ensure it's not excessively high or negative
        # Add a small epsilon to threshold to prevent division by zero if threshold is 0 (though slider min is 10)
        likelihood_ratio = (exceed_amount / (threshold + 1e-6)).clip(0, 1) # Cap the ratio at 1
        
        # Scale to 50-100% range
        breaching_pipelines_report['likelihood_percentage'] = likelihood_ratio * 50 + 50
        
        # Ensure it's rounded and handle any potential NaN from earlier operations
        breaching_pipelines_report['likelihood_percentage'] = breaching_pipelines_report['likelihood_percentage'].fillna(0).round(1)
    else:
        # If breaching_pipelines_report is empty, explicitly add the column as an empty Series
        # This ensures the column exists even if there's no data to populate it, preventing KeyError
        breaching_pipelines_report['likelihood_percentage'] = pd.Series(dtype='float64') 

    return breaching_pipelines_report

breaching_pipelines = predict_breaches(df_filtered, sla_threshold)

# --- Dashboard Content ---
tab1, tab2, tab3 = st.tabs(["Dashboard Overview", "Predicted Breaches Details", "Historical Data Viewer"])

with tab1:
    st.header("📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    total_runs = len(df_filtered)
    total_actual_breaches_in_range = df_filtered[df_filtered['duration_minutes'] > sla_threshold].shape[0]
    total_pipelines_predicted_to_breach = breaching_pipelines.shape[0]

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Total Pipeline Runs (Selected Range)</div>
            <div class="metric-value">{total_runs}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Actual Breaches In Range</div>
            <div class="metric-value">{total_actual_breaches_in_range}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Pipelines Predicted to Breach SLA</div>
            <div class="metric-value">{total_pipelines_predicted_to_breach}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Visualizations ---
    st.subheader("📈 Pipeline Performance Over Time")

    # Aggregate data for charting
    df_agg = df_filtered.copy()
    # Ensure 'date' column is created before grouping
    df_agg['date'] = df_agg['start_time'].dt.to_period('D').dt.to_timestamp()
    daily_avg_duration = df_agg.groupby('date')['duration_minutes'].mean().reset_index()

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=daily_avg_duration['date'], y=daily_avg_duration['duration_minutes'],
                                  mode='lines+markers', name='Average Duration', line=dict(color='#4A90E2'))) # Vibrant blue line
    fig_line.add_hline(y=sla_threshold, line_dash="dot", line_color="#E53935", annotation_text=f"SLA Threshold ({sla_threshold} min)", annotation_position="top right") # Clear red for breach
    fig_line.update_layout(
        title="Daily Average Pipeline Duration",
        xaxis_title="Date",
        yaxis_title="Average Duration (minutes)",
        hovermode="x unified",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)', # Transparent plot background for dark theme
        paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background
        font=dict(color="#F0F0F0") # Light text for chart labels
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("📊 Top 10 Pipelines by Average Duration")
    # CORRECTED: Use `name='avg_duration_minutes'` to correctly name the column
    top_10_pipelines = df_filtered.groupby('dag_id')['duration_minutes'].mean().nlargest(10).reset_index(name='avg_duration_minutes')

    fig_bar = go.Figure(go.Bar(
        x=top_10_pipelines['avg_duration_minutes'],
        y=top_10_pipelines['dag_id'],
        orientation='h',
        marker_color=['#E53935' if d > sla_threshold else '#4A90E2' for d in top_10_pipelines['avg_duration_minutes']] # Blue for good, clear red for breach
    ))
    fig_bar.update_layout(
        title="Average Duration by Pipeline (Top 10)",
        xaxis_title="Average Duration (minutes)",
        yaxis_title="Pipeline ID",
        yaxis={'autorange': "reversed"}, # Puts the highest average at the top
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#F0F0F0")
    )
    fig_bar.add_vline(x=sla_threshold, line_dash="dot", line_color="#E53935", annotation_text=f"SLA Threshold ({sla_threshold} min)", annotation_position="bottom right") # Clear red for breach
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- LLM-Generated Insights & Recommendations ---
    st.subheader("📝 AI-Generated Insights & Recommendations")
    # Call the LLM function with the breaching pipelines data
    llm_output = generate_llm_insight(breaching_pipelines, sla_threshold)
    st.markdown(
        f'<div class="llm-insight-box">{llm_output}</div>',
        unsafe_allow_html=True
    )

with tab2:
    st.header("📋 Predicted Breaches Details")
    if not breaching_pipelines.empty:
        st.write("The following pipelines are **predicted to breach SLA** based on their recent performance:")
        
        # --- NEW LOGIC: Only sort if DataFrame is not empty ---
        display_df = breaching_pipelines[['dag_id', 'last_run_duration_minutes', 'avg_duration_minutes', 'delay_minutes', 'likelihood_percentage', 'success']] \
                     .rename(columns={
                         'dag_id': 'Pipeline ID',
                         'last_run_duration_minutes': 'Last Run Duration (min)',
                         'avg_duration_minutes': 'Average Duration (min)',
                         'delay_minutes': 'Last Run Delay (min)',
                         'likelihood_percentage': 'Likelihood of Breach (%)',
                         'success': 'Last Run Status'
                     })
        
        # Apply sorting only if the DataFrame is not empty and the column exists (it will if not empty)
        if 'Likelihood of Breach (%)' in display_df.columns:
             display_df = display_df.sort_values(by='Likelihood of Breach (%)', ascending=False)
        
        st.dataframe(display_df.set_index('Pipeline ID'))
        # --- END NEW LOGIC ---

        st.info("Likelihood of Breach is a simplified estimation based on how much the pipeline's performance exceeds the SLA threshold.")
    else:
        st.info("🎉 No pipelines are currently predicted to breach the SLA. Good job!")

with tab3:
    st.header("🔍 Historical Data Viewer")
    st.write("View and explore the raw historical pipeline run data used for analysis.")
    st.dataframe(df_filtered) # Display the filtered data

# --- Clear All Button (in sidebar) ---
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear All Data & Reset App"):
    # Clear all items from session_state to reset the app
    # Only clear custom keys, not Streamlit internal ones
    keys_to_clear = [key for key in st.session_state.keys() if not key.startswith('__')]
    for key in keys_to_clear:
        del st.session_state[key]
    st.cache_data.clear() # Clear cached data as well
    st.rerun() # Rerun the app to clear the displays