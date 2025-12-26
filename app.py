# ===============================
# Smart AI Nutrition Application
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Smart AI Nutrition", layout="wide")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    encodings = ["utf-8-sig", "cp1256", "windows-1252", "latin1"]
    for enc in encodings:
        try:
            df = pd.read_csv("normalized.csv", encoding=enc)
            return df
        except UnicodeDecodeError:
            continue
    st.error("❌ Cannot read normalized.csv بسبب مشكلة ترميز")
    st.stop()

    for col in ["food_name", "name", "name_en", "Food", "food"]:
        if col in df.columns:
            df.rename(columns={col: "name_en"}, inplace=True)
            break

    if "name_en" not in df.columns:
        raise ValueError("Food name column not found")

    df.fillna(0, inplace=True)
    return df

df = load_data()

# -------------------------------
# Nutrients
# -------------------------------
NUTRIENTS = [
    "vitamin_c", "vitamin_b12", "vitamin_b6", "vitamin_a", "vitamin_d",
    "vitamin_e", "vitamin_k", "sodium", "potassium", "calcium",
    "magnesium", "selenium", "phosphorous", "manganese",
    "iron", "copper", "zinc", "folic_acid", "water"
]

AVAILABLE_NUTRIENTS = [n for n in NUTRIENTS if n in df.columns]

# -------------------------------
# Sidebar - Patient Info
# -------------------------------
st.sidebar.header("Patient Information")

patient_name = st.sidebar.text_input("Patient Name")

age_group = st.sidebar.selectbox(
    "Age Group",
    ["Child (4-12)", "Teen (13-18)", "Adult (19-50)", "Senior (50+)"]
)

patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

has_diabetes = st.sidebar.checkbox("Diabetic")
has_hypertension = st.sidebar.checkbox("Hypertensive")

# -------------------------------
# Diabetes Labs (ONLY if no deficiencies)
# -------------------------------
fasting = post_meal = random = hba1c = None
systolic = diastolic = None

# -------------------------------
# Deficient Nutrients
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Deficient Nutrients")

selected_nutrients = st.sidebar.multiselect(
    "Select nutrients with deficiency",
    AVAILABLE_NUTRIENTS
)

if has_diabetes and not selected_nutrients:
    st.sidebar.markdown("### Diabetes Lab Results")
    fasting = st.sidebar.number_input("Fasting Blood Sugar (mg/dL)", min_value=0.0)
    post_meal = st.sidebar.number_input("Postprandial Blood Sugar (mg/dL)", min_value=0.0)
    random = st.sidebar.number_input("Random Blood Sugar (mg/dL)", min_value=0.0)
    hba1c = st.sidebar.number_input("HbA1c (%)", min_value=0.0)

if has_hypertension and not selected_nutrients:
    st.sidebar.markdown("### Blood Pressure Readings")
    systolic = st.sidebar.number_input("Systolic Pressure", min_value=0.0)
    diastolic = st.sidebar.number_input("Diastolic Pressure", min_value=0.0)

patient_values = {}
if selected_nutrients:
    st.sidebar.markdown("### Lab Values")
    for n in selected_nutrients:
        patient_values[n] = st.sidebar.number_input(
            f"{n.replace('_',' ').title()}",
            min_value=0.0,
            step=0.1
        )

# -------------------------------
# Medical Nutrition Therapy Rules
# -------------------------------
def diabetes_foods(df):
    return df[df.get("sugars", 0) <= 10]

def hypertension_foods(df):
    return df[df.get("sodium", 0) <= 300]

# -------------------------------
# Generate Button
# -------------------------------
if st.button("Generate Nutrition Plan"):

    # ===============================
    # CASE 1: Diabetes / Hypertension ONLY
    # ===============================
    if (has_diabetes or has_hypertension) and not selected_nutrients:

        filtered_foods = df.copy()

        if has_diabetes:
            filtered_foods = diabetes_foods(filtered_foods)

        if has_hypertension:
            filtered_foods = hypertension_foods(filtered_foods)

        filtered_foods = filtered_foods.reset_index(drop=True)
        filtered_foods["Medical_Advice"] = "Medical Nutrition Therapy"

        st.session_state.filtered_foods = filtered_foods
        st.session_state.selected_nutrients = []
        st.session_state.medical_only = True
        st.session_state.page = 0

    # ===============================
    # CASE 2 & 3: Micronutrient Deficiency (AI)
    # ===============================
    else:
        if not patient_name or not selected_nutrients:
            st.warning("Please complete patient data")
            st.stop()

        filtered_foods = df.copy()

        X, y = [], []
        for _, row in filtered_foods.iterrows():
            vec = []
            for n in selected_nutrients:
                vec.append(row.get(n, 0))
                vec.append(patient_values.get(n, 0))
            X.append(vec)
            y.append(sum(vec))

        X = np.array(X)
        y = np.array(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(
            n_estimators=120,
            random_state=42
        )
        model.fit(X_scaled, y)

        filtered_foods["score"] = model.predict(X_scaled)

        filtered_foods = filtered_foods.sort_values(
            by="score", ascending=False
        ).reset_index(drop=True)

        st.session_state.filtered_foods = filtered_foods
        st.session_state.selected_nutrients = selected_nutrients
        st.session_state.medical_only = False
        st.session_state.page = 0

# -------------------------------
# Pagination Display (10 x 10)
# -------------------------------
st.markdown("---")
st.subheader("Recommended Foods")

if "filtered_foods" not in st.session_state:
    st.info("Generate nutrition plan to view results")
    st.stop()

df_show = st.session_state.filtered_foods
medical_only = st.session_state.medical_only

page_size = 10
total_rows = len(df_show)
max_page = (total_rows - 1) // page_size

start = st.session_state.page * page_size
end = start + page_size

display_cols = ["name_en","name_ar"]
if medical_only:
    display_cols.append("Medical_Advice")
else:
    display_cols += st.session_state.selected_nutrients

st.dataframe(
    df_show.loc[start:end-1, display_cols],
    use_container_width=True
)

st.markdown(
    f"**Showing {start + 1} – {min(end, total_rows)} of {total_rows} foods**"
)

col1, col2, col3 = st.columns([1,2,1])

with col1:
    if st.button("⬅ Previous") and st.session_state.page > 0:
        st.session_state.page -= 1

with col3:
    if st.button("Next ➡") and st.session_state.page < max_page:
        st.session_state.page += 1
