"""
Entry point of the Streamlit Application
Defines the structure of the sidebar menu and handles page navigation.
"""

import sys
from pathlib import Path

import streamlit as st

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    page_title="SecurityView",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# NAVIGATION
# =============================================================================
PAGES_PATH = BASE_DIR / "pages"

pages = [
    st.Page(PAGES_PATH / "visualisation.py", title="🏠 Visualisation"),
    st.Page(PAGES_PATH / "clustering.py", title="⚙️ Clustering"),
]

pg = st.navigation(pages)

# =============================================================================
# SIDEBAR FOOTER
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.caption("SecurityView v1.0")

# Run the selected page
pg.run()
