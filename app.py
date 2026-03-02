"""
Entry point of the Streamlit Application
Defines the structure of the sidebar menu and handles page navigation.
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from services.data_manager import DataManager

load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
ROOT_DIR = Path(__file__).parent
FILENAME = os.getenv("DATAFILE", "1h-attack-log.csv")

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="SecurityView",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# NAVIGATION
# =============================================================================
PAGES_PATH = ROOT_DIR / "pages"

pages = [
    st.Page(
        PAGES_PATH / "visualisation.py", 
        title="Visualisation",
        icon=":material/bar_chart:"
    ),
    st.Page(
        PAGES_PATH / "clustering.py", 
        title="Clustering",
        icon=":material/shapes:"
    ),
    st.Page(
        PAGES_PATH / "prediction.py", 
        title="Prédiction", 
        icon=":material/neurology:"
    ),
    st.Page(
        PAGES_PATH / "mcp.py", 
        title="MCP",
        icon=":material/chat:"
    ),
]

pg = st.navigation(pages)

# =============================================================================
# SIDEBAR FOOTER
# =============================================================================
# st.sidebar.caption("SecurityView v1.0")


# =============================================================================
# Instanciate global objects
# =============================================================================
st.session_state.data = DataManager(FILENAME)

# Run the selected page
pg.run()
