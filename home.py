import streamlit as st

st.set_page_config(page_title="Migraine Tracker", layout="wide")

st.title("Migraine Tracker App")
st.write("Use the sidebar to switch between pages:")
st.write("- **Tracker**: log daily entries")
st.write("- **Daily Log + Regression**: explore data and run the ordinal regression")

# cd /path/
# streamlit run home.py