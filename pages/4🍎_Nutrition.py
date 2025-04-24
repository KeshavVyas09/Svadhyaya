import streamlit as st
import google.generativeai as genai
from PIL import Image

# Set page title
st.set_page_config(page_title="SVADHYAYA: Nutrients Check 🍽️")

# Configure Google Gemini API
api_key = "AIzaSyA7bqotfmSNaCdN-jdOkPrQKqskENvsajY"
if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("Please enter your Google API Key.")

## Get response from Google Gemini Pro Vision API
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content([prompt, image[0]])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def input_image_setup(uploaded_file):
    # Return image data if file uploaded
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # File mime type
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        return None  # No file uploaded

## Initialize Streamlit app
st.title("Check Your Food: Protect Your Health 🍽️")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""  
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Tell me about The Food")

input_prompt="""
You are an expert nutritionist. Analyze the food items in the image and calculate the total calories. 
Provide a detailed breakdown of each food item and its estimated calorie count in the following format:

1. Item 1 - approximate number of calories
2. Item 2 - approximate number of calories
...

Then, provide the estimated total calorie count. If you are unsure of an exact calorie count, please provide the best possible estimate.
"""

## On submit button click
if submit:
    if uploaded_file is not None:
        image_data = input_image_setup(uploaded_file)
        if image_data:
            response = get_gemini_response("", image_data, input_prompt) # input_text removed, "" used
            st.subheader("The Response is:")
            st.write(response)
        else:
            st.error("Please upload an image.")
    else:
        st.error("Please upload an image.")
