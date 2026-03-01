"""
Entry point of the Streamlit Application
Defines the structure of the sidebar menu and handles page navigation.
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from services import Parser

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
    st.Page(PAGES_PATH / "visualisation.py", title="🏠 Visualisation"),
    st.Page(PAGES_PATH / "clustering.py", title="⚙️ Clustering"),
]

pg = st.navigation(pages)

# =============================================================================
# SIDEBAR FOOTER
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.caption("SecurityView v1.0")


# =============================================================================
# Instanciate global objects
# =============================================================================
st.session_state.parser = Parser(str(ROOT_DIR / "data" / FILENAME))


# Run the selected page
pg.run()
