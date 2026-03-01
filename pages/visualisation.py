import streamlit as st
from pandas import Timestamp

from services.charts import (
    access_distribution,
    deny_rate_distribution,
    deny_rules_distribution,
    deny_vs_permit_bubble,
    horizontal_vs_vertical_scan,
    requests_per_second_top,
    sensitive_ports_top,
    top_triggered_rules,
)

st.title("Visualisation")

data = st.session_state.data

# =============================================================================
# SIDEBAR — FILTERS
# =============================================================================
st.sidebar.header("Filtres")

t_min, t_max = data.time_range

start_date, end_date = st.sidebar.slider(
    "Fenêtre temporelle",
    min_value=t_min.to_pydatetime(),
    max_value=t_max.to_pydatetime(),
    value=(t_min.to_pydatetime(), t_max.to_pydatetime()),
    format="DD/MM/YY HH:mm",
)


@st.cache_data
def get_df(start: Timestamp, end: Timestamp):
    return st.session_state.data.get_filtered_df(start, end)


# Get the filtered dataframe based on the time window.
df = get_df(Timestamp(start_date), Timestamp(end_date))


# Summary metrics
total_ips = len(df)
high_deny = (df["deny_rate"] >= 0.8).sum()
st.sidebar.metric("IPs dans la fenêtre", total_ips)
st.sidebar.metric("IPs à fort deny rate (≥80%)", high_deny)

# =============================================================================
# TRAFFIC OVERVIEW
# =============================================================================

# Displaying the data
st.table(df.head())


st.header("Vue d'ensemble du trafic")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(access_distribution(df), width="stretch")
with col2:
    st.plotly_chart(deny_rate_distribution(df), width="stretch")

# =============================================================================
# SCANNING BEHAVIOUR
# =============================================================================
st.header("Comportements de scanning")

st.plotly_chart(horizontal_vs_vertical_scan(df), width="stretch")
st.plotly_chart(requests_per_second_top(df), width="stretch")

# =============================================================================
# DENY / PERMIT
# =============================================================================
st.header("Analyse Permit / Deny")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(deny_vs_permit_bubble(df), width="stretch")
with col2:
    st.plotly_chart(sensitive_ports_top(df), width="stretch")

# =============================================================================
# FIREWALL RULES
# =============================================================================
st.header("Règles pare-feu")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(top_triggered_rules(df), width="stretch")
with col2:
    st.plotly_chart(deny_rules_distribution(df), width="stretch")
