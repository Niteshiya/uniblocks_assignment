import streamlit as st
from pathlib import Path

# Page Config
st.set_page_config(page_title="Nitesh Kumar | Data Scientist", layout="centered",initial_sidebar_state='expanded')

# Load Resume File
resume_path = "data/NiteshKumar-April.pdf"
with open(resume_path, "rb") as f:
    resume_bytes = f.read()

# --- Header ---
st.title("Nitesh Kumar")
st.markdown("""
**Email:** [17niteshkumar17@gmail.com](mailto:17niteshkumar17@gmail.com)  
**Phone:** +91 79790 78551  
**Role:** *Advanced Analytics Lead | Data Scientist | Problem Solver*
""")

# --- Download Button ---
st.download_button(
    label="📄 Download My Resume",
    data=resume_bytes,
    file_name="Nitesh_Kumar_Resume.pdf",
    mime="application/pdf",
    help="Download the PDF version of my resume",
    use_container_width=True
)

asm_col1 , asm_col2 = st.columns(2)

with asm_col1:
    ml_assignment=st.button("ML Assignment",use_container_width=True,type='primary')
with asm_col2:
    pm_assignment = st.button("PM Assignment",use_container_width=True,type='primary',disabled=True)
    
if ml_assignment:
    st.switch_page("pages/1_ML Assignment.py")
if pm_assignment:
    st.switch_page("pages/2_PM Assignment.py")

# --- Summary ---
st.header("🔍 Professional Summary")
st.markdown("""
A skilled data scientist proficient in Python, AI, and analytics, known for delivering scalable, innovative solutions
that drive business value and product efficiency. Passionate about automating operations and personalizing education at scale.
""")

st.markdown(
    """
> **My AIM:** *“To be the very best, like no one ever was.”*  – Pokémon
""",
    unsafe_allow_html=True,
)


# --- Experience ---
st.header("💼 Experience")
with st.expander("Lead School – Advanced Analytics Lead"):
    st.markdown("""
- Built demand forecasting systems for millions of students and subjects.
- Designed "Book Scanner" in PASA app using image + text vector search.
- Created interactive AR-based content experiences.
    """)

with st.expander("Lead School – Data Scientist"):
    st.markdown("""
- Built Generative AI pipeline for dynamic lesson planning.
- Developed AI agent-based RAG question paper systems.
- Designed dashboards to reduce 95% ad-hoc analytics.
- Led large-scale school database scraping with segmentation.
    """)

with st.expander("Lead School – Product Analyst"):
    st.markdown("""
- Timetable creation via Genetic Algorithms.
- Automated worksheet evaluation using image & NLP.
- Replaced AWS Textract with custom marks entry tool.
    """)

with st.expander("Tredence Analytics – Analyst"):
    st.markdown("""
- Multi-objective optimization for marketing ROI.
- Modeled sports sponsorship returns using channel impressions.
    """)

# --- Skills ---
st.header("🧠 Skills")
cols = st.columns(2)

with cols[0]:
    st.subheader("Programming")
    st.markdown("""
- Python, C++, C#, CUDA, Embedded C
- HTML, Bash, LaTeX, Git, Vim
    """)

    st.subheader("Soft Skills")
    st.markdown("""
- Time Management, Problem-Solving
- Engaging Presentations, Documentation
- Leadership, On-site Coordination
    """)

with cols[1]:
    st.subheader("Data Science Tools")
    st.markdown("""
- Generative AI, PyTorch, TensorFlow
- Tableau, Power BI, Looker Studio, Excel
- Gradio, RAG Pipelines
    """)

# --- Education ---
st.header("🎓 Education")
st.markdown("""
- **B.Tech (Hons.) in Metallurgical and Materials Engineering**, NIT – 7.8 GPA  
- **Science Stream**, Sri Chaitanya Vishakapatnam – 93%
""")

# --- Interests ---
st.header("🌟 Life Beyond Work")
with st.expander("🍻 Beer Buddy Brainstormer"):
    st.markdown("""
    Creative thinker who enjoys ideating over a cold brew—turning casual talk into real solutions.
    """)
with st.expander("🎮 Gaming Strategist"):
    st.markdown("""
    Strategic gamer with a flair for tactical decision-making and team coordination. Loves Dota 2 & Catan.
    """)
with st.expander("🍜 Secret House Husband Chef"):
    st.markdown("""
    Weekend culinary explorer. Enjoys experimenting in the kitchen.
    """)
with st.expander("📺 Anime Aficionado"):
    st.markdown("""
    Passionate about anime with strong plots—favorites include Naruto, AoT, Vinland Saga, and OPM.
    """)

# --- Footer ---
st.markdown("---")
st.markdown("© 2025 Nitesh Kumar | Built with Streamlit ❤️")

