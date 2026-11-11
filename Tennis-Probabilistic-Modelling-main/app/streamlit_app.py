"""Streamlit entry point:  streamlit run app/streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title="ATP Edge", page_icon="🎾", layout="wide", initial_sidebar_state="expanded")

pages = [
    st.Page("views/board.py", title="Today's bets", icon="🎾", default=True),
    st.Page("views/slams.py", title="Grand Slam sets", icon="🏆", url_path="slams"),
    st.Page("views/match.py", title="Match detail", icon="🔎", url_path="match"),
    st.Page("views/performance.py", title="Results", icon="📈", url_path="performance"),
    st.Page("views/model.py", title="Model & test results", icon="🧠", url_path="model"),
    st.Page("views/guide.py", title="How it works", icon="📖", url_path="guide"),
]
st.navigation(pages).run()
