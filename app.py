# ===============================
# Smart AI Nutrition Application
# ===============================

import streamlit as st
import pandas as pd
import numpy as np

from hormone_engine import process_hormone
from hormone_config import HORMONE_CONFIG

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.pagesizes import A4

import arabic_reshaper
from bidi.algorithm import get_display

# ===============================
# Arabic Font
# ===============================
pdfmetrics.registerFont(TTFont('Arabic', 'Amiri-Regular.ttf'))

def fix_ar(text):
    if not text:
        return ""

    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Smart Nutrition AI",
    layout="wide"
)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():

    encodings = [
        "utf-8-sig",
        "cp1256",
        "windows-1252",
        "latin1"
    ]

    for enc in encodings:

        try:
            df = pd.read_csv(
                "normalized.csv",
                encoding=enc
            )

            break

        except UnicodeDecodeError:
            continue

    else:
        st.error("❌ Cannot read normalized.csv")
        st.stop()

    # ===============================
    # Rename Columns
    # ===============================
    for col in [
        "food_name",
        "name",
        "name_en",
        "Food",
        "food"
    ]:

        if col in df.columns:
            df.rename(
                columns={col: "name_en"},
                inplace=True
            )
            break

    if "name_en" not in df.columns:
        raise ValueError("Food name column not found")

    if "name_ar" not in df.columns:
        df["name_ar"] = ""

    df.fillna(0, inplace=True)

    return df

df = load_data()

# ===============================
# Nutrients
# ===============================
NUTRIENTS = [
    "vitamin_c",
    "vitamin_b12",
    "vitamin_b6",
    "vitamin_a",
    "vitamin_d",
    "vitamin_e",
    "vitamin_k",
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "selenium",
    "phosphorous",
    "manganese",
    "iron",
    "copper",
    "zinc",
    "folic_acid",
    "water"
]

AVAILABLE_NUTRIENTS = [
    n for n in NUTRIENTS
    if n in df.columns
]

# ===============================
# Sidebar
# ===============================
st.sidebar.header("Patient Information")

patient_name = st.sidebar.text_input(
    "Patient Name"
)

age_group = st.sidebar.selectbox(
    "Age Group",
    [
        "Infant (0-2)",
        "Child (3-12)",
        "Teen (13-18)",
        "Adult (19-50)",
        "Senior (50+)"
    ]
)

patient_gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

has_diabetes = st.sidebar.checkbox("Diabetic")
has_hypertension = st.sidebar.checkbox("Hypertensive")

# ===============================
# Nutrient Deficiencies
# ===============================
st.sidebar.markdown("---")
st.sidebar.header("Deficient Nutrients")

selected_nutrients = st.sidebar.multiselect(
    "Select nutrients with deficiency",
    AVAILABLE_NUTRIENTS
)

patient_values = {}

if selected_nutrients:

    st.sidebar.markdown("### Lab Values")

    for n in selected_nutrients:

        patient_values[n] = st.sidebar.number_input(
            f"{n.replace('_', ' ').title()}",
            min_value=0.0,
            step=0.1
        )

# ===============================
# Hormones
# ===============================
st.sidebar.markdown("---")
st.sidebar.header("Hormones $ Enzemes")

selected_hormones = st.sidebar.multiselect(
    "Select Hormones",
    list(HORMONE_CONFIG.keys())
)

hormone_values = {}

for h in selected_hormones:

    hormone_values[h] = st.sidebar.number_input(
        f"{h} Value",
        min_value=0.0,
        step=0.1
    )

# ===============================
# Diabetes Labs
# ===============================
fasting = None
post_meal = None
random = None
hba1c = None

if has_diabetes:

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Diabetes Lab Results")

    fasting = st.sidebar.number_input(
        "Fasting Blood Sugar",
        min_value=0.0
    )

    post_meal = st.sidebar.number_input(
        "Postprandial Blood Sugar",
        min_value=0.0
    )

    random = st.sidebar.number_input(
        "Random Blood Sugar",
        min_value=0.0
    )

    hba1c = st.sidebar.number_input(
        "HbA1c %",
        min_value=0.0
    )

# ===============================
# Blood Pressure
# ===============================
systolic = None
diastolic = None

if has_hypertension:

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Blood Pressure")

    systolic = st.sidebar.number_input(
        "Systolic Pressure",
        min_value=0.0
    )

    diastolic = st.sidebar.number_input(
        "Diastolic Pressure",
        min_value=0.0
    )

# ===============================
# Medical Filters
# ===============================
def diabetes_foods(df):

    if "sugars" in df.columns:
        return df[df["sugars"] <= 10]

    return df

def hypertension_foods(df):

    if "sodium" in df.columns:
        return df[df["sodium"] <= 300]

    return df

# ===============================
# Patient Summary
# ===============================
def generate_patient_summary():

    summary = []

    # Diseases
    if has_diabetes and has_hypertension:
        summary.append(
            "Patient has Diabetes and Hypertension."
        )

    elif has_diabetes:
        summary.append(
            "Patient has Diabetes."
        )

    elif has_hypertension:
        summary.append(
            "Patient has Hypertension."
        )

    else:
        summary.append(
            "Patient has no chronic diseases."
        )

    # Nutrients
    if selected_nutrients:

        deficiency_text = ", ".join([
            n.replace("_", " ").title()
            for n in selected_nutrients
        ])

        summary.append(
            f"Nutrient deficiencies detected: {deficiency_text}."
        )

    # Hormones
    if selected_hormones:

        hormone_text = ", ".join(selected_hormones)

        summary.append(
            f"Hormone tests selected: {hormone_text}."
        )

    return " ".join(summary)

# ===============================
# Generate Plan
# ===============================
if st.button("Generate Nutrition Plan"):

    st.session_state.patient_summary = generate_patient_summary()

    st.session_state.nutrient_foods = {}
    st.session_state.hormone_foods = {}

    filtered_foods = df.copy()

    # ===============================
    # Disease Filtering
    # ===============================
    if has_diabetes:
        filtered_foods = diabetes_foods(filtered_foods)

    if has_hypertension:
        filtered_foods = hypertension_foods(filtered_foods)

    # ===============================
    # Nutrient Processing
    # ===============================
    for nutrient in selected_nutrients:

        nutrient_df = filtered_foods.copy()

        if nutrient in nutrient_df.columns:

            nutrient_df = nutrient_df[
                nutrient_df[nutrient] > 0
            ]

            nutrient_df = nutrient_df.sort_values(
                by=nutrient,
                ascending=False
            ).reset_index(drop=True)

            st.session_state.nutrient_foods[
                nutrient
            ] = nutrient_df

    # ===============================
    # Hormone Processing
    # ===============================
    for hormone in selected_hormones:

        value = hormone_values.get(hormone, 0)

        config = HORMONE_CONFIG[hormone]

        foods, status, message, advice = process_hormone(
            filtered_foods,
            value,
            config,
            age_group,
            patient_gender
        )

        st.info(f"{hormone}: {message}")
        st.success(advice)

        st.session_state.hormone_foods[
            hormone
        ] = foods.reset_index(drop=True)

# ===============================
# Results
# ===============================
st.markdown("---")
st.subheader("Recommended Foods")

# ===============================
# Patient Summary
# ===============================
if "patient_summary" in st.session_state:

    st.markdown("### 🧾 Patient Summary")

    st.info(
        st.session_state.patient_summary
    )

# ===============================
# Nutrient Results
# ===============================
if "nutrient_foods" in st.session_state:

    for nutrient, foods in st.session_state.nutrient_foods.items():

        st.markdown(
            f"## 🧪 {nutrient.replace('_', ' ').title()}"
        )

        page_key = f"page_{nutrient}"

        if page_key not in st.session_state:
            st.session_state[page_key] = 0

        page_size = 10

        total_rows = len(foods)

        max_page = max(
            (total_rows - 1) // page_size,
            0
        )

        start = st.session_state[page_key] * page_size
        end = start + page_size

        cols = [
            "name_en",
            "name_ar",
            nutrient
        ]

        cols = [
            c for c in cols
            if c in foods.columns
        ]

        st.dataframe(
            foods.loc[start:end-1, cols],
            use_container_width=True
        )

        col1, col2, col3 = st.columns([1,2,1])

        with col1:

            if st.button(
                "⬅ Previous",
                key=f"prev_{nutrient}"
            ):

                if st.session_state[page_key] > 0:
                    st.session_state[page_key] -= 1

        with col3:

            if st.button(
                "Next ➡",
                key=f"next_{nutrient}"
            ):

                if st.session_state[page_key] < max_page:
                    st.session_state[page_key] += 1

# ===============================
# Hormone Results
# ===============================
if "hormone_foods" in st.session_state:

    st.markdown("---")
    st.subheader("🧬 Hormones $ Enzymes Recommendations")

    for hormone, foods in st.session_state.hormone_foods.items():

        st.markdown(f"## 🍽️ {hormone}")

        page_key = f"page_hormone_{hormone}"

        if page_key not in st.session_state:
            st.session_state[page_key] = 0

        page_size = 10

        total_rows = len(foods)

        max_page = max(
            (total_rows - 1) // page_size,
            0
        )

        start = st.session_state[page_key] * page_size
        end = start + page_size

        cols = [
            "name_en",
            "name_ar"
        ]

        st.dataframe(
            foods.loc[start:end-1, cols],
            use_container_width=True
        )

        col1, col2, col3 = st.columns([1,2,1])

        with col1:

            if st.button(
                "⬅ Previous",
                key=f"prev_hormone_{hormone}"
            ):

                if st.session_state[page_key] > 0:
                    st.session_state[page_key] -= 1

        with col3:

            if st.button(
                "Next ➡",
                key=f"next_hormone_{hormone}"
            ):

                if st.session_state[page_key] < max_page:
                    st.session_state[page_key] += 1

# ===============================
# PDF Generator
# ===============================
def generate_pdf():

    doc = SimpleDocTemplate(
        "nutrition_report.pdf",
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "title",
        parent=styles["Title"],
        alignment=1
    )

    section_style = ParagraphStyle(
        "section",
        parent=styles["Heading2"],
        textColor=colors.darkblue
    )

    normal_style = styles["Normal"]

    arabic_style = ParagraphStyle(
        name='arabic',
        parent=styles['Normal'],
        fontName='Arabic',
        alignment=2
    )

    elements = []

    # ===============================
    # Title
    # ===============================
    elements.append(
        Paragraph(
            f"Nutrition Report - {patient_name}",
            title_style
        )
    )

    elements.append(Spacer(1, 10))

    # ===============================
    # Summary
    # ===============================
    elements.append(
        Paragraph(
            "Patient Summary",
            section_style
        )
    )

    elements.append(
        Paragraph(
            generate_patient_summary(),
            normal_style
        )
    )

    elements.append(Spacer(1, 15))

    # ===============================
    # Nutrient Tables
    # ===============================
    if "nutrient_foods" in st.session_state:

        for nutrient, foods in st.session_state.nutrient_foods.items():

            elements.append(
                Paragraph(
                    f"{nutrient} Foods",
                    section_style
                )
            )

            table_data = [
                ["Food EN", "Food AR"]
            ]

            top_foods = foods.head(10)

            for _, row in top_foods.iterrows():

                table_data.append([
                    Paragraph(
                        str(row.get("name_en", "")),
                        normal_style
                    ),

                    Paragraph(
                        fix_ar(row.get("name_ar", "")),
                        arabic_style
                    )
                ])

            table = Table(
                table_data,
                colWidths=[220, 220]
            )

            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
                ("GRID", (0,0), (-1,-1), 1, colors.grey),
                ("PADDING", (0,0), (-1,-1), 5)
            ]))

            elements.append(table)
            elements.append(Spacer(1, 15))

    # ===============================
    # Hormone Tables
    # ===============================
    if "hormone_foods" in st.session_state:

        for hormone, foods in st.session_state.hormone_foods.items():

            elements.append(
                Paragraph(
                    f"{hormone} Foods",
                    section_style
                )
            )

            table_data = [
                ["Food EN", "Food AR"]
            ]

            top_foods = foods.head(10)

            for _, row in top_foods.iterrows():

                table_data.append([
                    Paragraph(
                        str(row.get("name_en", "")),
                        normal_style
                    ),

                    Paragraph(
                        fix_ar(row.get("name_ar", "")),
                        arabic_style
                    )
                ])

            table = Table(
                table_data,
                colWidths=[220, 220]
            )

            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightgreen),
                ("GRID", (0,0), (-1,-1), 1, colors.grey),
                ("PADDING", (0,0), (-1,-1), 5)
            ]))

            elements.append(table)
            elements.append(Spacer(1, 15))

    doc.build(elements)

    with open(
        "nutrition_report.pdf",
        "rb"
    ) as f:

        return f.read()

# ===============================
# Export PDF
# ===============================
if (
    "nutrient_foods" in st.session_state
    or
    "hormone_foods" in st.session_state
):

    st.markdown("---")
    st.subheader("📄 Export Report")

    if st.button("Generate Professional PDF Report"):

        pdf = generate_pdf()

        st.download_button(
            label="⬇ Download PDF",
            data=pdf,
            file_name=f"{patient_name}_nutrition_report.pdf",
            mime="application/pdf"
        )