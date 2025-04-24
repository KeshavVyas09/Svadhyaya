import os
from datetime import datetime
import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.common import DayOfWeek
from utils.scheduler import create_schedule, update_schedule, suggest_habits_based_on_method


# Config
GOOGLE_API_KEY = "AIzaSyA7bqotfmSNaCdN-jdOkPrQKqskENvsajY"
GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/1VH1N_1LNhtwWX1tV70Y68_Vhi-9-U7oNNKN1wbkIm0M/export?format=csv"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Load habits data from Google Sheets
csv_url = GOOGLE_SHEETS_URL.replace("/edit#gid=", "/export?format=csv&gid=")
habits_df = pd.read_csv(csv_url)

# Initialize session state variables
if "todo_list" not in st.session_state:
    st.session_state.todo_list = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Set page title and icon
st.set_page_config(
    page_title="HabitByBit",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/2611.png",
)

st.markdown("<h1 style='text-align: center'>HabitByBit</h1>", unsafe_allow_html=True)

st.subheader("Your Schedule:")
st.dataframe(habits_df)

# Image slider for popular methods
st.subheader("Popular-Methods:")

image_files = [
    "Habitbybit/m-1.jpeg",
    "Habitbybit/m-2.jpg",
    "Habitbybit/m-3.jpg",
    "Habitbybit/m-4.jpg",
    "Habitbybit/m-5.jpg",
]

# Initialize image index for slider
if "image_index" not in st.session_state:
    st.session_state.image_index = 0

# Layout columns for image slider buttons and image
col1, col2, col3 = st.columns([1,6,1])

with col1:
    if st.button("⬅"):
        if st.session_state.image_index > 0:
            st.session_state.image_index -= 1

with col3:
    if st.button("➡"):
        if st.session_state.image_index < len(image_files) - 1:
            st.session_state.image_index += 1

with col2:
    st.image(image_files[st.session_state.image_index], use_container_width=True)

# Select day for schedule
selected_day = st.selectbox(
    "Day of the week for your schedule",
    [e.value for e in DayOfWeek],
    index=datetime.now().weekday(),
)

# Generate schedule button
if st.button("Generate Schedule"):
    st.session_state.todo_list = create_schedule(
        habits_df,
        DayOfWeek[selected_day.upper()]
    )

# Display schedule with checkboxes
if st.session_state.todo_list:
    st.subheader(f"{selected_day} Schedule")
    cols = st.columns(2)
    for i, todo in enumerate(st.session_state.todo_list):
        col = cols[i % 2]
        col.checkbox(str(todo), key=f"todo_{i}")

    # Input to modify schedule
    question = st.text_input(
        "Modify your schedule:",
        key="schedule_question",
        placeholder="e.g., Move my workout to after lunch",
    )
    if question:
        st.session_state.todo_list = update_schedule(
            st.session_state.todo_list,
            question
        )
else:
    st.info("Your schedule is empty. Generate a schedule to get started.")

st.markdown("---")  # Separator

# Chat with Gemini for habit strategies
with st.expander("Chat with Gemini for Habit Strategies", expanded=True):
    user_input = st.chat_input("Ask Gemini about habits or methods like Pomodoro, Kaizen...")

    if user_input:
        chat_gpt = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        chat_chain = LLMChain(llm=chat_gpt, prompt=PromptTemplate(
            input_variables=["query"],
            template="You're a habit-building expert. Answer or suggest strategies based on: {query}"
        ))

        response = chat_chain.run(query=user_input)
        st.session_state.chat_history.append((user_input, response))

    # Show chat history
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
