import streamlit as st
import datetime

# Optional: Set Streamlit page-wide settings
st.set_page_config(
    page_title="SVADHYAYA: Health Eval Using AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('## 🌿Welcome to SVADHYAYA: Health Evaluation Using AI')

# Display image responsively by setting width to 80% of the container width
st.image("static/svadhyaya.png", use_container_width=True)

# Footer
st.markdown(
    f"<div style='text-align: center; color: gray;'>"
    f"Developed with 💚 by Keshav Vyas, Hema Thelwani & Shreya Kukreti | "
    f"{datetime.datetime.now().year} © SVADHYAYA"
    f"</div>", unsafe_allow_html=True
)
