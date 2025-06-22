# ⏰ SLA Breach Predictor Dashboard

## Table of Contents
1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Design & User Interface](#design--user-interface)
4.  [Screenshots](#screenshots)
5.  [Technologies Used](#technologies-used)
6.  [Setup and Installation](#setup-and-installation)
    * [Prerequisites](#prerequisites)
    * [Clone the Repository](#clone-the-repository)
    * [Virtual Environment (Recommended)](#virtual-environment-recommended)
    * [Install Dependencies](#install-dependencies)
    * [Google Gemini API Key Setup](#google-gemini-api-key-setup)
7.  [How to Run the Application](#how-to-run-the-application)
8.  [Usage](#usage)
9.  [Project Structure](#project-structure)
10. [Future Enhancements](#future-enhancements)
11. [Contributing](#contributing)

## Introduction
The **SLA Breach Predictor Dashboard** is a powerful Streamlit application designed to help data engineering teams monitor and proactively manage Service Level Agreements (SLAs) for their data pipelines. By analyzing historical pipeline run data, the dashboard identifies potential SLA breaches and provides AI-driven insights and actionable recommendations to prevent future occurrences.

This tool aims to move from reactive troubleshooting to proactive problem-solving, ensuring data delivery consistency and reliability.

## Features
* **Flexible Data Input:** Upload your own historical pipeline run data via CSV, or use a pre-loaded sample dataset for demonstration.
* **Configurable SLA Threshold:** Dynamically adjust the SLA threshold (in minutes) to simulate different performance targets.
* **Dashboard Overview:** Get a high-level summary of total pipeline runs, actual breaches in the selected range, and pipelines predicted to breach.
* **Interactive Visualizations:**
    * **Pipeline Performance Over Time:** A line chart showing daily average pipeline duration to spot trends.
    * **Top Pipelines by Average Duration:** A bar chart highlighting pipelines that frequently exceed or are close to the SLA threshold.
* **AI-Generated Insights & Recommendations:** Leverage the Google Gemini large language model to receive natural language summaries of breach situations, potential root causes, and actionable mitigation strategies.
* **Predicted Breaches Details:** A detailed table of pipelines predicted to breach, including their last run duration, average duration, and a simplified "likelihood of breach" percentage.
* **Historical Data Viewer:** Explore the raw historical data used for analysis.
* **Dynamic Date Range Selection:** Filter data for analysis based on a customizable date range.

## Design & User Interface
The dashboard features a **sleek dark mode design** for improved readability and a modern aesthetic, especially useful during long monitoring sessions. The professional color palette ensures clarity and focus on critical metrics and insights.

## Screenshots
*(Replace this section with actual screenshots of your dashboard in action once you run it)*

![Dashboard Overview - Dark Mode](path/to/your/screenshot_overview.png)
![Predicted Breaches - Dark Mode](path/to/your/screenshot_details.png)
![AI Insights - Dark Mode](path/to/your/screenshot_ai_insights.png)

## Technologies Used
* **Python 3.8+**
* **Streamlit:** For building interactive web applications.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations, especially in sample data generation.
* **Plotly:** For interactive data visualizations.
* **Google Generative AI (`google-generativeai`):** For powering the AI-driven insights.
* **python-dotenv:** For managing environment variables (API keys).

## Setup and Installation

### Prerequisites
* Python 3.8 or higher installed.
* `pip` (Python package installer).
* Git (for cloning the repository).

### Clone the Repository
First, clone this repository to your local machine:
```bash
git clone [https://github.com/your-username/sla-breach-predictor.git](https://github.com/your-username/sla-breach-predictor.git)
cd sla-breach-predictor
````

### Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Install Dependencies

All required Python packages are listed in `requirements.txt`. Install them using pip:

```bash
pip install -r requirements.txt
```

### Google Gemini API Key Setup

The AI insights feature requires a Google Gemini API key.

1.  Go to the [Google AI Studio](https://aistudio.google.com/app/apikey) to generate an API key.
2.  Create a file named `.env` in the root directory of your project (same level as `main.py`).
3.  Add your API key to this file in the following format:
    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
    Replace `"YOUR_API_KEY_HERE"` with your actual Gemini API key.
    **Note:** Ensure your API key has access to `gemini-1.5-flash` or `gemini-1.5-pro` models, as `gemini-1.0-pro-vision` has been deprecated. The `ai_logic.py` automatically tries to find a suitable model.

## How to Run the Application

Once you have completed the setup, activate your virtual environment and run the Streamlit application:

```bash
streamlit run main.py
```

This command will open the dashboard in your default web browser (usually at `http://localhost:8501`).

## Usage

1.  **Data Input:** Use the sidebar to either upload your own CSV file or proceed with the default sample data.
2.  **Time Range & SLA Threshold:** Adjust the date range and the SLA threshold slider in the sidebar to configure your analysis parameters.
3.  **Navigate Tabs:**
      * **Dashboard Overview:** See high-level metrics and key visualizations.
      * **Predicted Breaches Details:** View a table of pipelines identified as potential breach risks.
      * **Historical Data Viewer:** Inspect the raw data.
4.  **AI Insights:** Review the AI-generated summary and recommendations in the "Dashboard Overview" tab for actionable advice.
5.  **Reset:** Use the "Clear All Data & Reset App" button in the sidebar to clear session data and start fresh.

## Project Structure

```
sla-breach-predictor/
├── main.py               # Main Streamlit application file with UI and data display
├── ai_logic.py           # Contains the AI model initialization and insight generation logic
├── requirements.txt      # List of Python dependencies
├── .env                  # Stores environment variables (e.g., GEMINI_API_KEY)
└── README.md             # This file
```

## Future Enhancements

  * **Advanced Prediction Models:** Integrate more sophisticated machine learning models (e.g., Prophet, XGBoost) for more accurate breach forecasting.
  * **Root Cause Analysis:** Enhance AI insights with more granular data analysis for deeper root cause identification.
  * **Real-time Data Integration:** Connect to live data sources (e.g., Airflow, Snowflake, BigQuery) instead of CSV uploads.
  * **User Authentication:** Implement user login for multi-user environments.
  * **Notification System:** Add alerts for predicted breaches (e.g., email, Slack).
  * **Customizable Dashboards:** Allow users to save and load custom dashboard configurations.
  * **Deployment Automation:** Scripts for easier deployment to cloud platforms (e.g., Streamlit Community Cloud, Heroku).

## Contributing

Contributions are welcome\! If you have suggestions for improvements or find any issues, please open an issue or submit a pull request.
