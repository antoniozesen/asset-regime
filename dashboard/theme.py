import streamlit as st

DARK_COLORS = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "text": "#e6edf3",
    "muted": "#8b949e",
    "grid": "#30363d",
    "blue": "#58a6ff",
    "green": "#3fb950",
    "red": "#f85149",
    "amber": "#d29922",
    "violet": "#bc8cff",
}


def apply_theme() -> None:
    st.markdown(
        f"""
<style>
body, .stApp {{ background-color: {DARK_COLORS['bg']}; color: {DARK_COLORS['text']}; }}
.stSelectbox, .stMultiSelect, .stTextInput {{ background-color: {DARK_COLORS['panel']}; }}
div[data-testid="stMetricValue"] {{ font-family: monospace; font-size: 20px; }}
</style>
""",
        unsafe_allow_html=True,
    )
