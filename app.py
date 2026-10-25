"""Run with: python -m streamlit run app.py"""
import hashlib
import importlib
import io
import pandas as pd
import streamlit as st
from coconut import CoconutWorkbench

st.set_page_config(page_title="Coconut Production", page_icon="🥥", layout="wide")

SCREENS = {
    "1 · Data quality": "views.data_quality",
    "2 · Explore production": "views.explore",
    "3 · Monthly prediction": "views.monthly",
    "4 · Annual forecast": "views.annual",
    "5 · Other data mining": "views.mining",
    "6 · Management dashboard": "views.dashboard",
    "7 · Download results": "views.downloads",
}


def upload_sidebar():
    st.sidebar.title("🥥 Coconut production")
    st.sidebar.caption("Upload once, then run the modules you need.")
    uploaded = st.sidebar.file_uploader("Upload coconut data", type=["xlsx", "csv"], key="upload")
    if uploaded is not None:
        content = uploaded.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        sheet = 0
        usable = True
        if uploaded.name.lower().endswith(".xlsx"):
            try:
                if st.session_state.get("sheet_digest") != digest:
                    with pd.ExcelFile(io.BytesIO(content), engine="openpyxl") as book:
                        st.session_state["sheet_names"] = list(book.sheet_names)
                    st.session_state["sheet_digest"] = digest
                sheet = st.sidebar.selectbox("Worksheet", st.session_state["sheet_names"], key="sheet")
            except Exception as exc:
                usable = False
                st.sidebar.error(f"Cannot read this Excel workbook: {exc}")
        if st.sidebar.button("Load and clean data", type="primary", disabled=not usable, key="load"):
            try:
                candidate = CoconutWorkbench()
                candidate.load_bytes(content, uploaded.name, sheet_name=sheet)
            except Exception as exc:
                st.sidebar.error(f"Data were not loaded: {exc}")
            else:
                st.session_state["workbench"] = candidate
                for key in ["decision_scenario", "scenario_signature", "export_bytes", "task_runs"]:
                    st.session_state.pop(key, None)
                st.session_state["loaded_signature"] = (digest, str(sheet))
                st.sidebar.success("Data cleaned. Choose a module below.")
        if "workbench" in st.session_state and st.session_state.get("loaded_signature") != (digest, str(sheet)):
            st.sidebar.info("A different file or worksheet is selected. Click Load and clean data to use it.")
    wb = st.session_state.get("workbench")
    if wb is not None:
        st.sidebar.caption(f"Loaded: {wb.source_name}\n\nLatest month: {wb.audit['last_month']}")
    return wb


def main():
    wb = upload_sidebar()
    selected = st.sidebar.radio("Module", list(SCREENS), key="module")
    st.sidebar.caption("Analyses remain available while this session is open. Reloading data resets model results.")
    if wb is None:
        st.title("Coconut production decision support")
        st.write("Upload an Excel workbook or CSV using the left panel and click **Load and clean data**.")
        st.info("Start with data quality and exploration. Run monthly or annual models when needed, "
                "then open the management dashboard.")
        with st.expander("Expected columns"):
            st.code("Year, Month, Barangay, Harvested Area (ha), Harvest (kg)\n"
                    "Optional: Rainfall (mm), Temperature (°C), prices, Inflation Rate (%), Weather")
            st.write("Use one record per barangay and month. Month may be a name, abbreviation, or number.")
        return
    from views.common import source_note
    source_note(wb)
    importlib.import_module(SCREENS[selected]).render(wb)


if __name__ == "__main__":
    main()
