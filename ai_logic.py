# ai_logic.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def initialize_gemini_model():
    """
    Initializes the Google Gemini model using the API key from environment variables.
    It dynamically finds an available text generation model, prioritizing newer ones.
    Returns the model object or None if an error occurs.
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None, "GEMINI_API_KEY not found in environment variables. Please set it in your .env file."
        genai.configure(api_key=api_key)

        model_name = None
        available_models = list(genai.list_models())

        # Prioritize models in this order, based on the error message suggestion and best practices
        # 'gemini-1.5-flash' is recommended for speed and cost-effectiveness for text
        # 'gemini-1.5-pro' is a more capable, larger model
        # 'gemini-pro' is the original general-purpose text model
        preferred_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']

        for preferred in preferred_models:
            for m in available_models:
                # Check if the preferred name is part of the model's actual name
                # and if it supports the 'generateContent' method.
                if preferred in m.name and 'generateContent' in m.supported_generation_methods:
                    # Also, try to exclude explicit 'vision' models unless explicitly needed
                    if 'vision' not in m.name.lower():
                        model_name = m.name
                        break # Found a suitable preferred model, break out of inner loop
            if model_name: # Found a preferred model, break out of outer loop
                break
        
        # Fallback: if no explicitly preferred model found, try any other model with generateContent
        if not model_name:
            for m in available_models:
                if 'generateContent' in m.supported_generation_methods:
                    # As a last resort, pick any non-vision, non-deprecated model
                    if 'vision' not in m.name.lower() and '1.0-pro' not in m.name.lower() and 'deprecated' not in m.name.lower():
                        model_name = m.name
                        break
            
        if not model_name:
            return None, "No suitable Gemini model found with 'generateContent' capability. Please check model availability for your region/API key and consider updating your 'google-generativeai' library."

        model = genai.GenerativeModel(model_name)
        return model, None # Return model and no error
    except Exception as e:
        return None, f"Error initializing Gemini model: {e}. The AI service might be unavailable or rate-limited. Please ensure your GEMINI_API_KEY is correct and retry."

def generate_llm_insight(breach_df, sla_threshold_minutes):
    """
    Generates a natural language insight and recommendations for SLA breaches
    using the Gemini LLM.

    Args:
        breach_df (pd.DataFrame): DataFrame containing pipelines predicted to breach.
        sla_threshold_minutes (int): The defined SLA threshold in minutes.

    Returns:
        str: AI-generated summary and recommendations, or an error message if an issue occurs.
    """
    model, init_error = initialize_gemini_model()
    if init_error:
        return f"AI insights are currently unavailable: {init_error}" # Propagate initialization error

    if breach_df.empty:
        return "No significant SLA breaches predicted based on the provided data and threshold. Keep up the good work!"

    # Prepare data for the prompt
    total_breaches = len(breach_df)
    unique_pipelines_breaching = breach_df['dag_id'].nunique()

    # Safely get top_breaching_pipelines_counts for prompt
    if not breach_df.empty:
        top_breaching_pipelines_counts = breach_df['dag_id'].value_counts().nlargest(3)
        top_breaching_pipelines_str = ", ".join([f"{name} ({int(count)} breaches)" for name, count in top_breaching_pipelines_counts.items()])
    else:
        top_breaching_pipelines_str = "N/A"

    avg_delay_breaching = breach_df['delay_minutes'].mean() if not breach_df.empty else 0
    max_delay_breaching = breach_df['delay_minutes'].max() if not breach_df.empty else 0

    # Construct the prompt for the LLM
    prompt_text = f"""
    Analyze the following data about predicted SLA breaches for data pipelines.
    The defined SLA threshold is {sla_threshold_minutes} minutes.

    **Summary of Breaches:**
    - Total predicted breaches: {total_breaches}
    - Unique pipelines predicted to breach: {unique_pipelines_breaching}
    - Top 3 pipelines with most predicted breaches: {top_breaching_pipelines_str}.
    - Average predicted delay for breaching pipelines: {avg_delay_breaching:.1f} minutes.
    - Maximum predicted delay for a single breach: {max_delay_breaching:.1f} minutes.

    Based on this data, provide:
    1.  A concise summary of the overall SLA breach situation.
    2.  Potential common root causes for these types of data pipeline SLA breaches (e.g., resource contention, data volume spikes, inefficient code, upstream dependencies, seasonal loads).
    3.  Actionable mitigation strategies and recommendations for the engineering team to prevent future breaches.
    4.  Phrase the output in a clear, professional, and slightly urgent tone if there are many high-risk breaches, or encouraging if few.
    """

    try:
        response = model.generate_content(prompt_text)
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "AI returned an empty response. This might indicate an issue with the prompt or the model's ability to generate content."
    except Exception as e:
        return f"Error generating AI insights: {e}. The AI service might be unavailable or rate-limited. Please ensure your GEMINI_API_KEY is correct and retry."