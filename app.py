# ===============================
# Smart AI Nutrition Application
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import arabic_reshaper
from bidi.algorithm import get_display

pdfmetrics.registerFont(TTFont('Arabic', 'Amiri-Regular.ttf'))
def fix_ar(text):
    if not text:
        return ""
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)
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
    ["Infant (0-2)","Child (3-12)", "Teen (13-18)", "Adult (19-50)", "Senior (50+)"]
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
# ===============================
# Patient Summary Function
# ===============================
def generate_patient_summary():
    summary = []

    if has_diabetes and has_hypertension:
        summary.append("Patient has Diabetes and Hypertension.")
    elif has_diabetes:
        summary.append("Patient has Diabetes.")
    elif has_hypertension:
        summary.append("Patient has Hypertension.")
    else:
        summary.append("Patient has no chronic diseases.")

    if selected_nutrients:
        deficiency_text = ", ".join(
            [n.replace("_", " ").title() for n in selected_nutrients]
        )
        summary.append(f"Nutrient deficiencies detected: {deficiency_text}.")
    else:
        summary.append("No micronutrient deficiencies.")

    return " ".join(summary)


# ===============================
# PDF Imports (برا أي دالة)
# ===============================
# -------------------------------
# Generate Button
# -------------------------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4

def generate_pdf(data):

    doc = SimpleDocTemplate("nutrition_report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()

    # ===============================
    # Styles
    # ===============================
    title_style = ParagraphStyle(
        'title',
        parent=styles['Title'],
        alignment=1,
        spaceAfter=10
    )

    section_style = ParagraphStyle(
        'section',
        parent=styles['Heading2'],
        textColor=colors.darkblue,
        spaceAfter=6
    )

    normal_style = styles['Normal']

    elements = []

    # ===============================
    # Title
    # ===============================
    elements.append(Paragraph(f"Nutrition Report - {patient_name}", title_style))
    elements.append(Spacer(1, 10))

    # ===============================
    # Conditions Fix
    # ===============================
    conditions = []
    if has_diabetes:
        conditions.append("Diabetes")
    if has_hypertension:
        conditions.append("Hypertension")

    condition_text = ", ".join(conditions) if conditions else "None"

    # ===============================
    # Patient Info Table
    # ===============================
    patient_data = [
        ["Name", patient_name],
        ["Age Group", age_group],
        ["Gender", patient_gender],
        ["Conditions", condition_text]
    ]

    table = Table(patient_data, colWidths=[120, 250])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("BOX", (0,0), (-1,-1), 1, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 15))

    # ===============================
    # Summary
    # ===============================
    elements.append(Paragraph("Patient Summary", section_style))
    elements.append(Paragraph(generate_patient_summary(), normal_style))
    elements.append(Spacer(1, 15))

    # ===============================
    # AI Note
    # ===============================
    elements.append(Paragraph(
        "This recommendation is generated based on AI-assisted nutritional analysis.",
        styles['Italic']
    ))
    elements.append(Spacer(1, 15))

    # ===============================
    # Deficiencies
    # ===============================
    if selected_nutrients:

        for nutrient in selected_nutrients:

            elements.append(Paragraph(
                f"{nutrient.replace('_',' ').title()} Deficiency",
                section_style
            ))

            df_nutrient = data[data[nutrient] > 0].copy()
            df_nutrient = df_nutrient.sort_values(by=nutrient, ascending=False)

            elements.append(Paragraph("Top Recommended Foods:", normal_style))
            elements.append(Spacer(1, 5))

            table_data = [["Food (EN)", "Food (AR)", "Value"]]

            top_n = df_nutrient.head(min(8, len(df_nutrient)))
            for _, row in top_n.iterrows():

               table_data.append([
            Paragraph(str(row.get("name_en", "")), styles['Normal']),

            Paragraph(fix_ar(row.get("name_ar", "")),
            ParagraphStyle(
                name='arabic_table',
                parent=styles['Normal'],
                fontName='Arabic',
                alignment=2
            )
        ),

             Paragraph(str(round(row.get(nutrient, 0), 2)), styles['Normal'])
    ])

        

            table = Table(table_data, colWidths=[150, 150, 80])

            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                ("PADDING", (0,0), (-1,-1), 5),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 10))

            elements.append(Paragraph(
                f"It is recommended to increase dietary intake of {nutrient.replace('_',' ')}-rich foods to correct the deficiency.",
                normal_style
            ))

            elements.append(Spacer(1, 15))

    else:
        elements.append(Paragraph("General Recommendation", section_style))
        elements.append(Paragraph(
            "Follow a balanced diet suitable for your medical condition.",
            normal_style
        ))

    doc.build(elements)

    with open("nutrition_report.pdf", "rb") as f:
        return f.read()
if st.button("Generate Nutrition Plan"):

    st.session_state.patient_summary = generate_patient_summary()
    filtered_foods = df.copy()

    # ===============================
    # ✅ فلترة حسب الحالة المرضية (دايمًا)
    # ===============================
    if has_diabetes:
        filtered_foods = diabetes_foods(filtered_foods)

    if has_hypertension:
        filtered_foods = hypertension_foods(filtered_foods)

    # ===============================
    # CASE 1: Medical Only
    # ===============================
    if (has_diabetes or has_hypertension) and not selected_nutrients:

        filtered_foods = filtered_foods.reset_index(drop=True)
        filtered_foods["Medical_Advice"] = "Medical Nutrition Therapy"

        st.session_state.filtered_foods = filtered_foods
        st.session_state.selected_nutrients = []
        st.session_state.medical_only = True

    # ===============================
    # CASE 2: Deficiency + AI
    # ===============================
    else:
        if not patient_name or not selected_nutrients:
            st.warning("Please complete patient data")
            st.stop()

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
# ===============================
# Recommended Foods Display
# ===============================
st.markdown("---")
st.subheader("Recommended Foods")

if "patient_summary" in st.session_state:
    st.markdown("### 🧾 Patient Summary")
    st.info(st.session_state.patient_summary)

if "filtered_foods" not in st.session_state:
    st.info("Generate nutrition plan to view results")
    st.stop()

df_show = st.session_state.filtered_foods
medical_only = st.session_state.medical_only
selected_nutrients = st.session_state.selected_nutrients

page_size = 10

# ===============================
# CASE 1: Medical Only
# ===============================
if medical_only:
    if "page_medical" not in st.session_state:
        st.session_state.page_medical = 0

    total_rows = len(df_show)
    max_page = (total_rows - 1) // page_size

    start = st.session_state.page_medical * page_size
    end = start + page_size

    st.dataframe(
        df_show.loc[start:end-1, ["name_en", "name_ar", "Medical_Advice"]],
        use_container_width=True
    )

    st.markdown(
        f"**Showing {start + 1} – {min(end, total_rows)} of {total_rows} foods**"
    )

    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        if st.button("⬅ Previous", key="prev_med") and st.session_state.page_medical > 0:
            st.session_state.page_medical -= 1

    with col3:
        if st.button("Next ➡", key="next_med") and st.session_state.page_medical < max_page:
            st.session_state.page_medical += 1


# ===============================
# CASE 2: لكل عنصر Pagination مستقل
# ===============================
else:
    for nutrient in selected_nutrients:

        st.markdown(f"### 🧪 Foods for {nutrient.replace('_',' ').title()} Deficiency")

        page_key = f"page_{nutrient}"
        if page_key not in st.session_state:
            st.session_state[page_key] = 0

        df_nutrient = df_show[df_show[nutrient] > 0].copy()

        df_nutrient = df_nutrient.sort_values(
            by=nutrient, ascending=False
        ).reset_index(drop=True)

        total_rows = len(df_nutrient)
        max_page = (total_rows - 1) // page_size

        start = st.session_state[page_key] * page_size
        end = start + page_size

        cols_to_show = ["name_en", "name_ar", nutrient]

        st.dataframe(
            df_nutrient.loc[start:end-1, cols_to_show],
            use_container_width=True
        )

        st.markdown(
            f"**Showing {start + 1} – {min(end, total_rows)} of {total_rows} foods**"
        )

        col1, col2, col3 = st.columns([1,2,1])

        with col1:
            if st.button("⬅ Previous", key=f"prev_{nutrient}") and st.session_state[page_key] > 0:
                st.session_state[page_key] -= 1

        with col3:
            if st.button("Next ➡", key=f"next_{nutrient}") and st.session_state[page_key] < max_page:
                st.session_state[page_key] += 1

        st.markdown("---")

# ===============================
# 📄 Export Report
# ===============================
if "filtered_foods" in st.session_state:

    st.markdown("### 📄 Export Report")

    if st.button("Generate Professional PDF Report"):

        pdf = generate_pdf(st.session_state.filtered_foods)

        st.download_button(
            label="⬇ Download PDF",
            data=pdf,
            file_name=f"{patient_name}_nutrition_report.pdf",
            mime="application/pdf"
        )