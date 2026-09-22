import streamlit as st

pg = st.navigation([
    st.Page(r"E:\my_projects\NRC_MODEL_DEV\test\pages\Home.py", title="Home", icon=":material/home:"),
    st.Page(r"E:\my_projects\NRC_MODEL_DEV\test\pages\page_1.py", title="About me", icon="🧾"),
])

pg.run()