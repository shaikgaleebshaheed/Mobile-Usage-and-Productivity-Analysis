import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Mobile Usage and Productivity Analysis",
    page_icon="📱",
    layout="wide"
)

# Load trained model and encoder
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "best_model.pkl")
encoder = joblib.load(BASE_DIR / "label_encoder.pkl")

# Dashboard title
st.title("📱 Mobile Usage and Productivity Analysis")
st.write(
    "Analyze your daily smartphone usage and its relationship with productivity."
)
st.caption(f"Algorithm Used: {type(model).__name__}")

# Sidebar inputs
st.sidebar.header("Enter Your Daily Usage")

screen_time = st.sidebar.slider(
    "Daily Screen Time (hours)", 0.0, 16.0, 6.0, 0.5
)
social_media = st.sidebar.slider(
    "Social Media (hours)", 0.0, 12.0, 2.0, 0.5
)
gaming = st.sidebar.slider(
    "Gaming (hours)", 0.0, 12.0, 1.0, 0.5
)
sleep = st.sidebar.slider(
    "Sleep (hours)", 0.0, 16.0, 8.0, 0.5
)
work_study = st.sidebar.slider(
    "Work/Study (hours)", 0.0, 16.0, 5.0, 0.5
)

# Usage summary
st.subheader("📊 Your Usage Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Screen Time", f"{screen_time:.1f} hrs")
col2.metric("Social Media", f"{social_media:.1f} hrs")
col3.metric("Gaming", f"{gaming:.1f} hrs")

# Usage chart
chart_data = pd.DataFrame(
    {
        "Hours": [
            screen_time,
            social_media,
            gaming,
            sleep,
            work_study
        ]
    },
    index=[
        "Screen Time",
        "Social Media",
        "Gaming",
        "Sleep",
        "Work/Study"
    ]
)

st.bar_chart(chart_data, horizontal=True)

# Analyze button
if st.button("Analyze My Usage", type="primary"):

    # Prepare input for the Random Forest model
    input_data = pd.DataFrame([{
        "daily_screen_time_hours": screen_time,
        "social_media_hours": social_media,
        "gaming_hours": gaming,
        "sleep_hours": sleep,
        "work_study_hours": work_study
    }])

    # Actual model prediction
    prediction = model.predict(input_data)[0]
    result = encoder.inverse_transform([int(prediction)])[0]

    # Reference rule used to generate training labels
    high_risk = (
        screen_time >= 8 and
        (social_media + gaming >= 4 or sleep < 6)
    )
    rule_result = "Yes" if high_risk else "No"

    # Display model result
    st.subheader("Analysis Result")

    if result == "Yes":
        st.warning("⚠️ Higher-risk usage pattern predicted by the model.")
        st.write(
            "Consider reviewing your screen time, entertainment habits, "
            "and sleep schedule."
        )
    else:
        st.success("✅ Lower-risk usage pattern predicted by the model.")
        st.write(
            "The model classified your entered usage pattern as lower-risk."
        )

    # Compare prediction with training rule
    with st.expander("View prediction details"):
        st.write("**Random Forest prediction:**", result)
        st.write("**Reference rule result:**", rule_result)

        if result != rule_result:
            st.info(
                "The model prediction differs from the reference rule. "
                "The trained model may not reproduce the rule perfectly."
            )
        else:
            st.write("The model prediction matches the reference rule.")

    # Detailed usage
    st.subheader("📋 Detailed Usage")
    st.dataframe(chart_data, use_container_width=True)

    st.caption(
        "Educational project only. The dataset uses synthetic, rule-based "
        "labels. This output is not a medical diagnosis or a validated "
        "measure of smartphone addiction."
    )