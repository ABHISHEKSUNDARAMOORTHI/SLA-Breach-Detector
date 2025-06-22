import streamlit as st

def apply_custom_css():
    """Applies custom CSS for a dark theme and consistent styling."""
    st.markdown(
        """
        <style>
        /* Base Dark Theme Colors */
        :root {
            --background-color: #0F1117; /* Very dark blue */
            --primary-text-color: #E6EDF3; /* Light gray/white */
            --accent-blue: #58A6FF;
            --accent-blue-light: #63b3ed;
            --accent-blue-dark: #4299e1;
            --success-green: #3FB950;
            --warning-yellow: #DD9F1B;
            --danger-red: #FF4D4F;
            --border-color: #4a5568; /* Box borders/separators */
            --input-background: #22252B; /* Background for text inputs, file uploaders */
            --container-background-light: #1A1D24; /* Slightly lighter background for specific containers */
        }

        /* General Body Styling */
        body {
            color: var(--primary-text-color);
            background-color: var(--background-color);
            font-family: 'Segoe UI', 'Roboto', sans-serif; /* Modern, clean font */
        }

        /* Streamlit Overrides */
        /* Main container background */
        .stApp {
            background-color: var(--background-color);
            color: var(--primary-text-color);
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-text-color);
            font-weight: 600; /* Slightly bolder headers */
        }

        /* Buttons */
        .stButton>button {
            background-color: var(--accent-blue);
            color: var(--primary-text-color);
            border: none;
            padding: 0.75rem 1.25rem;
            border-radius: 0.5rem;
            font-weight: 600;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .stButton>button:hover {
            background-color: var(--accent-blue-light);
            transform: translateY(-2px); /* Slight lift on hover */
        }
        .stButton>button:active {
            background-color: var(--accent-blue-dark);
            transform: translateY(0);
        }

        /* Input fields (text, number, date) */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stDateInput>div>div>input,
        .stFileUploader>section {
            background-color: var(--input-background);
            color: var(--primary-text-color);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 0.5rem 0.75rem;
        }

        /* Selectbox/Dropdowns */
        .stSelectbox>div>div {
            background-color: var(--input-background);
            color: var(--primary-text-color);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
        }
        .stSelectbox>div>div:focus-within {
            border-color: var(--accent-blue); /* Highlight on focus */
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: var(--input-background);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            color: var(--primary-text-color) !important;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem; /* Space below header */
        }
        .streamlit-expanderContent {
            background-color: var(--background-color); /* Matches app background */
            border: 1px solid var(--border-color);
            border-top: none; /* No top border to connect to header visually */
            border-bottom-left-radius: 0.5rem;
            border-bottom-right-radius: 0.5rem;
            padding: 1rem;
        }

        /* Containers & Boxes for structure */
        .stContainer {
            background-color: var(--background-color); /* Default Streamlit container is transparent */
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            margin-bottom: 1.5rem; /* Spacing between sections */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow */
        }

        /* Specific box for LLM insights */
        .llm-insight-box {
            background-color: var(--container-background-light); /* Slightly lighter for distinction */
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Dataframes */
        .stDataFrame {
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
        }
        /* Custom styling for dataframe headers and rows if needed
           Note: Streamlit's dataframe styling is somewhat restrictive via CSS alone.
           For complex table styling, consider using st.table and creating HTML. */
        .dataframe th {
            background-color: var(--input-background);
            color: var(--primary-text-color);
        }
        .dataframe tr:nth-child(even) {
            background-color: #1a1d24; /* Slightly different for alternating rows */
        }

        /* Metrics */
        .stMetric > div > label { /* Label of the metric */
            color: var(--primary-text-color);
        }
        .stMetric > div > div { /* Value of the metric */
            color: var(--accent-blue); /* Default value color */
            font-size: 2.5rem; /* Larger font for key metrics */
            font-weight: 700;
        }
        /* Metric indicators (delta) */
        .stMetric > div > div > div > svg {
             fill: var(--success-green); /* Default for up arrow */
        }
        .stMetric > div > div > div > svg[data-testid="stDeltaIconDown"] {
             fill: var(--danger-red); /* For down arrow */
        }

        /* Specific styling for risk indicators */
        .risk-indicator {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 0.75rem;
            font-weight: bold;
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 1rem;
            min-width: 120px; /* Ensure a minimum width */
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        .risk-indicator.high {
            background-color: var(--danger-red);
            color: var(--primary-text-color); /* Light text on red */
        }
        .risk-indicator.medium {
            background-color: var(--warning-yellow);
            color: var(--background-color); /* Dark text on yellow for contrast */
        }
        .risk-indicator.low {
            background-color: var(--success-green);
            color: var(--primary-text-color); /* Light text on green */
        }

        /* General Markdown styling within st.markdown(unsafe_allow_html=True) */
        .st-emotion-cache-nahz7x { /* Targeting the element containing markdown */
            color: var(--primary-text-color);
        }
        .st-emotion-cache-1629p8f { /* Further targeting markdown elements */
            color: var(--primary-text-color);
        }

        /* Custom styling for alerts (st.error, st.info, st.success, st.warning) */
        .stAlert {
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .stAlert.error {
            background-color: rgba(255, 77, 79, 0.15); /* Light red background */
            color: var(--danger-red);
            border-left: 5px solid var(--danger-red);
        }
        .stAlert.info {
            background-color: rgba(88, 166, 255, 0.15); /* Light blue background */
            color: var(--accent-blue);
            border-left: 5px solid var(--accent-blue);
        }
        .stAlert.success {
            background-color: rgba(63, 185, 80, 0.15); /* Light green background */
            color: var(--success-green);
            border-left: 5px solid var(--success-green);
        }
        .stAlert.warning {
            background-color: rgba(221, 159, 27, 0.15); /* Light yellow background */
            color: var(--warning-yellow);
            border-left: 5px solid var(--warning-yellow);
        }

        /* Streamlit's internal layout paddings (to reduce white space if desired) */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

def set_page_config():
    """Sets the Streamlit page configuration."""
    st.set_page_config(
        page_title="SLA Breach Predictor",
        page_icon="🔮", # You can choose an emoji or a path to an image
        layout="wide", # Use a wide layout
        initial_sidebar_state="auto" # or "collapsed"
    )

def display_header():
    """Displays the main header and project description."""
    st.title("🔮 SLA Breach Predictor")
    st.markdown(
        """
        Predict whether your data pipelines are likely to breach their Service Level Agreements (SLAs)
        based on historical performance, delays, and seasonal trends. Get proactive insights and
        recommendations to optimize your data operations.
        """
    )
    st.markdown("---") # Visual separator

def display_footer():
    """Displays the footer with project credits."""
    st.markdown("---")
    st.caption("🚀 SLA Breach Predictor | Powered by Streamlit & Google Gemini API")