import streamlit as st
from PIL import Image

# Page config
st.set_page_config(page_title="SVADHYAYA - Home", page_icon="🌿", layout="centered")

# Title and welcome
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>Welcome to SVADHYAYA 🌿</h1>", unsafe_allow_html=True)
st.info("SMALL STEPS EVERY DAY.")
st.markdown("---")

# Introduction Section
st.header("About SVADHYAYA")
st.markdown("""
Hey there 👋 I’m SVADHYAYA.. your personal AI-powered wellness buddy.

I’m here to help you build a healthier, more balanced life, one step at a time.
With the power of Artificial Intelligence.. I bring you simple, effective tools for yoga, habit tracking and mindful nutrition - all in one place. Whether you’re just starting your journey or looking to grow deeper, I’ve got your back with guidance, motivation and support.

Let’s grow together - your wellness journey starts with me!
""")

# Display project intro video using Streamlit's st.video with file path
try:
    st.video("static/Svadhyaya-intro.mp4", start_time=0)
except Exception:
    pass

st.markdown("---")

# Features Section
st.header("Key Features")

st.info("1. Yoga Pose Analysis -- Keshav Vyas")
st.markdown("""
To get the most out of the Yoga Pose Analysis feature:
- Stand in a well-lit area with your full body visible to your device's camera.
- Follow the on-screen instructions to perform yoga poses.
- Use the real-time feedback to adjust your posture and improve accuracy.
- Practice regularly to track your progress and enhance your form.
""")

st.info("2. HabitByBit - Habit Tracking and Scheduling -- Shreya Kukreti")
st.markdown("""
Maximize your habit-building with HabitByBit by:
- Importing your existing habits from Google Sheets for seamless integration.
- Reviewing your personalized daily schedule generated by Gemini AI each morning.
- Applying Kaizen and Pomodoro techniques as suggested to build habits gradually.
- Using natural language commands to easily update or modify your schedule anytime.
""")

st.info("3. Nutrition Awareness -- Hema Thelwani")
st.markdown("""
Make the most of the Nutrition Awareness feature by:
- Exploring nutrient information and healthy eating tips provided within the app.
- Logging your meals regularly to track your nutritional intake.
- Using personalized recommendations to adjust your diet for better health.
- Staying consistent and mindful of your eating habits to support your wellness journey.
""")

