import streamlit as st

pg = st.navigation([
    st.Page(r"app_branch\pages\Home.py", title="Home", icon=":material/home:"),
    st.Page(r"app_branch\pages\page_1.py", title="About me", icon="🧾"),
])

pg.run()