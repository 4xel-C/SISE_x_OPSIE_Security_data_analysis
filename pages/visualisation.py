import streamlit as st

st.title("Visualisation")

# load datamanager
df = st.session_state.data.df

st.write(df)
