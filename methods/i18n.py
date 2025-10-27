import streamlit as st
import json
from methods.logger import logger

# --- Internationalization (i18n) ---

@st.cache_data(ttl=3600)
def _load_translations(language: str) -> dict:
    """Loads the translation file for the selected language."""
    try:
        with open(f"locales/{language}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.log(f"Translation file for language '{language}' not found. Defaulting to German.", severity=1)
        with open("locales/de.json", "r", encoding="utf-8") as f:
            return json.load(f)

def t(key: str, **kwargs):
    """Returns the translated string for a given key."""
    lang = st.session_state.get("lang", "de")
    translations = _load_translations(lang)
    return translations.get(key, key).format(**kwargs)

