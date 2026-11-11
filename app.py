import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from pathlib import Path

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# PATH
# ============================================================

# Mengambil lokasi folder tempat app.py berada
BASE_DIR = Path(__file__).resolve().parent

# Lokasi model dan dataset
MODEL_PATH = BASE_DIR / "model" / "dropout_prediction_pipeline.pkl"
DATA_PATH = BASE_DIR / "data" / "dashboard_data.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# ============================================================
# MAPPING LABEL
# ============================================================

MARITAL_MAP = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto union",
    6: "Legally separated"
}

GENDER_MAP = {
    0: "Female",
    1: "Male"
}

BINARY_MAP = {
    0: "No",
    1: "Yes"
}

ATTENDANCE_MAP = {
    0: "Evening",
    1: "Daytime"
}

PREVIOUS_MAP = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle or equivalent",
    38: "Basic education 2nd cycle or equivalent",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)"
}

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATA DASHBOARD
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# ============================================================
# CEK FILE
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error(
        "File model tidak ditemukan. Pastikan "
        "'model/dropout_prediction_pipeline.pkl' tersedia."
    )
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(
        "File dashboard_data.csv tidak ditemukan. Pastikan "
        "'data/dashboard_data.csv' tersedia."
    )
    st.stop()

model = load_model()
df = load_data()

# ============================================================
# HEADER
# ============================================================

st.title("🎓 Student Dropout Prediction System")
st.subheader("Sistem Monitoring dan Prediksi Risiko Dropout Mahasiswa")

st.write(
    """
    Aplikasi ini membantu pihak Jaya Jaya Institut dalam memahami
    kondisi mahasiswa berdasarkan data akademik, demografis,
    dan kondisi pendukung lainnya.

    Aplikasi terdiri dari dua bagian utama:
    - 📊 **Dashboard** untuk memantau kondisi mahasiswa.
    - 🤖 **Prediksi Risiko Dropout** untuk memprediksi apakah mahasiswa
      berpotensi **Dropout atau Graduate**.
    """
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Navigasi")

menu = st.sidebar.radio(
    "Pilih halaman:",
    [
        "📊 Dashboard",
        "🤖 Prediksi Risiko Dropout"
    ]
)

# ============================================================
# DASHBOARD
# ============================================================

if menu == "📊 Dashboard":

    st.header("📊 Dashboard Mahasiswa")

    st.write(
        """
        Dashboard memberikan gambaran deskriptif mengenai status mahasiswa
        dan karakteristik yang berkaitan dengan kondisi akademik,
        demografis, serta faktor pendukung lainnya.
        """
    )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    st.sidebar.divider()
    st.sidebar.subheader("🔎 Filter Dashboard")

    status_options = [
        status for status in ["Dropout", "Enrolled", "Graduate"]
        if status in df["Status"].dropna().unique()
    ]

    selected_status = st.sidebar.multiselect(
        "Status Mahasiswa",
        options=status_options,
        default=status_options
    )

    if not selected_status:
        st.warning("Pilih minimal satu status mahasiswa.")
        st.stop()

    filtered_df = df[df["Status"].isin(selected_status)].copy()

    # --------------------------------------------------------
    # AGE GROUP
    # --------------------------------------------------------

    filtered_df["Age_group"] = pd.cut(
        filtered_df["Age_at_enrollment"],
        bins=[0, 20, 25, 30, 40, 100],
        labels=["≤20", "21–25", "26–30", "31–40", ">40"],
        include_lowest=True
    )

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    total_students = len(filtered_df)

    dropout_students = len(
        filtered_df[filtered_df["Status"] == "Dropout"]
    )

    graduate_students = len(
        filtered_df[filtered_df["Status"] == "Graduate"]
    )

    enrolled_students = len(
        filtered_df[filtered_df["Status"] == "Enrolled"]
    )

    finished_students = dropout_students + graduate_students

    if finished_students > 0:
        dropout_rate = dropout_students / finished_students * 100
    else:
        dropout_rate = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Total Mahasiswa",
        f"{total_students:,}"
    )

    col2.metric(
        "🔴 Dropout",
        f"{dropout_students:,}"
    )

    col3.metric(
        "🟢 Graduate",
        f"{graduate_students:,}"
    )

    col4.metric(
        "📉 Dropout Rate",
        f"{dropout_rate:.2f}%"
    )

    if enrolled_students > 0:
        st.caption(
            f"Mahasiswa Enrolled: {enrolled_students:,}. "
            "Status Enrolled belum digunakan sebagai target training model."
        )

    st.divider()

    # --------------------------------------------------------
    # 1. STATUS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("1. Distribusi Status Mahasiswa")

    status_order = ["Dropout", "Enrolled", "Graduate"]

    status_count = (
        filtered_df["Status"]
        .value_counts()
        .reindex(status_order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(status_count.index, status_count.values)

    ax.set_title("Distribusi Status Mahasiswa")
    ax.set_xlabel("Status Mahasiswa")
    ax.set_ylabel("Jumlah Mahasiswa")

    for i, value in enumerate(status_count.values):
        ax.text(
            i,
            value,
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # 2. AGE VS STATUS
    # --------------------------------------------------------

    st.subheader("2. Distribusi Usia Berdasarkan Status")

    fig, ax = plt.subplots(figsize=(10, 5))

    available_status = [
        status for status in status_order
        if status in filtered_df["Status"].unique()
    ]

    age_data = [
        filtered_df.loc[
            filtered_df["Status"] == status,
            "Age_at_enrollment"
        ].dropna()
        for status in available_status
    ]

    ax.boxplot(
        age_data,
        tick_labels=available_status,
        patch_artist=True
    )

    ax.set_title("Distribusi Usia Saat Pendaftaran Berdasarkan Status")
    ax.set_xlabel("Status Mahasiswa")
    ax.set_ylabel("Usia Saat Pendaftaran")

    st.pyplot(fig)
    plt.close(fig)


    # --------------------------------------------------------
    # 3. AGE GROUP VS DROPOUT RATE
    # --------------------------------------------------------

    st.subheader("3. Dropout Rate Berdasarkan Kelompok Usia")

    age_status = pd.crosstab(
        filtered_df["Age_group"],
        filtered_df["Status"]
    )

    if "Dropout" in age_status.columns:
        finished_by_age = (
            age_status.get("Dropout", 0)
            + age_status.get("Graduate", 0)
        )

        age_dropout_rate = (
            age_status["Dropout"] / finished_by_age.replace(0, pd.NA) * 100
        )

        age_dropout_rate = age_dropout_rate.fillna(0)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            age_dropout_rate.index.astype(str),
            age_dropout_rate.values
        )

        ax.set_title("Dropout Rate Berdasarkan Kelompok Usia")
        ax.set_xlabel("Kelompok Usia")
        ax.set_ylabel("Dropout Rate (%)")

        for i, value in enumerate(age_dropout_rate.values):
            ax.text(
                i,
                value,
                f"{value:.1f}%",
                ha="center",
                va="bottom"
            )

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Data Dropout tidak tersedia pada filter yang dipilih.")

    # --------------------------------------------------------
    # 4. COURSE VS STATUS
    # --------------------------------------------------------

    st.subheader("4. Status Mahasiswa Berdasarkan Program Studi")

    course_status = pd.crosstab(
        filtered_df["Course"],
        filtered_df["Status"]
    )

    for status in status_order:
        if status not in course_status.columns:
            course_status[status] = 0

    course_status = course_status[status_order]

    course_status = (
        course_status
        .sort_values("Dropout", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    course_status.plot(
        kind="bar",
        stacked=False,
        ax=ax
    )

    ax.set_title(
        "Top 10 Course Code Berdasarkan Jumlah Dropout"
    )
    ax.set_xlabel("Course Code")
    ax.set_ylabel("Jumlah Mahasiswa")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Status")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Course ditampilkan sebagai kode numerik karena pemetaan nama "
        "program studi tidak tersedia pada data dashboard."
    )

    # --------------------------------------------------------
    # 5. TUITION
    # --------------------------------------------------------

    st.subheader(
        "5. Status Mahasiswa Berdasarkan Kondisi Pembayaran"
    )

    tuition_col = "Tuition_fees_up_to_date_label"

    if tuition_col in filtered_df.columns:

        tuition_status = pd.crosstab(
            filtered_df[tuition_col],
            filtered_df["Status"]
        )

        for status in status_order:
            if status not in tuition_status.columns:
                tuition_status[status] = 0

        tuition_status = tuition_status[status_order]

        fig, ax = plt.subplots(figsize=(10, 5))

        tuition_status.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Status Mahasiswa Berdasarkan Pembayaran Tuition Fees"
        )
        ax.set_xlabel("Status Pembayaran")
        ax.set_ylabel("Jumlah Mahasiswa")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Status Mahasiswa")

        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------
    # 6. SCHOLARSHIP
    # --------------------------------------------------------

    st.subheader(
        "6. Status Mahasiswa Berdasarkan Beasiswa"
    )

    scholarship_col = "Scholarship_holder_label"

    if scholarship_col in filtered_df.columns:

        scholarship_status = pd.crosstab(
            filtered_df[scholarship_col],
            filtered_df["Status"]
        )

        for status in status_order:
            if status not in scholarship_status.columns:
                scholarship_status[status] = 0

        scholarship_status = scholarship_status[status_order]

        fig, ax = plt.subplots(figsize=(10, 5))

        scholarship_status.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Status Mahasiswa Berdasarkan Kepemilikan Beasiswa"
        )
        ax.set_xlabel("Status Beasiswa")
        ax.set_ylabel("Jumlah Mahasiswa")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Status Mahasiswa")

        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------
    # 7. GENDER
    # --------------------------------------------------------

    st.subheader(
        "7. Status Mahasiswa Berdasarkan Gender"
    )

    gender_col = "Gender_label"

    if gender_col in filtered_df.columns:

        gender_status = pd.crosstab(
            filtered_df[gender_col],
            filtered_df["Status"]
        )

        for status in status_order:
            if status not in gender_status.columns:
                gender_status[status] = 0

        gender_status = gender_status[status_order]

        fig, ax = plt.subplots(figsize=(10, 5))

        gender_status.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Status Mahasiswa Berdasarkan Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Jumlah Mahasiswa")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Status Mahasiswa")

        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------
    # 8. SEMESTER 1
    # --------------------------------------------------------

    st.subheader("8. Performa Akademik Semester 1")

    semester1_summary = (
        filtered_df
        .groupby("Status")[
            [
                "Curricular_units_1st_sem_approved",
                "Curricular_units_1st_sem_grade"
            ]
        ]
        .mean()
        .reindex(status_order)
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))

        semester1_summary[
            "Curricular_units_1st_sem_approved"
        ].plot(kind="bar", ax=ax)

        ax.set_title(
            "Rata-rata Mata Kuliah Lulus Semester 1"
        )
        ax.set_xlabel("Status")
        ax.set_ylabel("Rata-rata Mata Kuliah Lulus")
        ax.tick_params(axis="x", rotation=0)

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))

        semester1_summary[
            "Curricular_units_1st_sem_grade"
        ].plot(kind="bar", ax=ax)

        ax.set_title("Rata-rata Nilai Semester 1")
        ax.set_xlabel("Status")
        ax.set_ylabel("Rata-rata Nilai")
        ax.tick_params(axis="x", rotation=0)

        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------
    # 9. SEMESTER 2
    # --------------------------------------------------------

    st.subheader("9. Performa Akademik Semester 2")

    semester2_summary = (
        filtered_df
        .groupby("Status")[
            [
                "Curricular_units_2nd_sem_approved",
                "Curricular_units_2nd_sem_grade"
            ]
        ]
        .mean()
        .reindex(status_order)
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))

        semester2_summary[
            "Curricular_units_2nd_sem_approved"
        ].plot(kind="bar", ax=ax)

        ax.set_title(
            "Rata-rata Mata Kuliah Lulus Semester 2"
        )
        ax.set_xlabel("Status")
        ax.set_ylabel("Rata-rata Mata Kuliah Lulus")
        ax.tick_params(axis="x", rotation=0)

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))

        semester2_summary[
            "Curricular_units_2nd_sem_grade"
        ].plot(kind="bar", ax=ax)

        ax.set_title("Rata-rata Nilai Semester 2")
        ax.set_xlabel("Status")
        ax.set_ylabel("Rata-rata Nilai")
        ax.tick_params(axis="x", rotation=0)

        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------
    # 10. INSIGHT
    # --------------------------------------------------------

    st.divider()
    st.subheader("💡 Insight Utama")

    st.markdown(
        """
        Berdasarkan analisis deskriptif pada data historis:

        - **Performa akademik** merupakan aspek penting yang perlu dipantau,
          terutama jumlah mata kuliah yang berhasil diselesaikan dan nilai
          pada semester 1 dan semester 2.
        - **Usia saat pendaftaran** dapat digunakan untuk melihat perbedaan
          karakteristik mahasiswa antar-status.
        - **Kondisi pembayaran tuition fees** perlu diperhatikan sebagai
          informasi pendukung dalam monitoring mahasiswa.
        - **Kepemilikan beasiswa** dapat menjadi informasi tambahan dalam
          memahami kondisi mahasiswa.
        - **Perbedaan status antar course** dapat membantu institusi
          menentukan prioritas monitoring.
        """
    )

    st.info(
        "Catatan: hubungan pada dashboard bersifat deskriptif/asosiatif "
        "dan tidak menunjukkan hubungan sebab-akibat."
    )

# ============================================================
# PREDIKSI DROPOUT
# ============================================================

elif menu == "🤖 Prediksi Risiko Dropout":

    st.header("🤖 Prediksi Risiko Dropout")

    st.write(
        """
        Masukkan informasi mahasiswa pada form berikut.

        Model hanya memiliki **dua kelas target**:
        - **0 = Graduate**
        - **1 = Dropout**

        Status **Enrolled tidak digunakan sebagai target training**, karena
        status tersebut belum merupakan label akhir. Model ini digunakan
        untuk membantu mengidentifikasi pola yang lebih dekat dengan
        **Graduate atau Dropout** berdasarkan data yang dimasukkan.
        """
    )

    st.warning(
        "Hasil model merupakan prediksi berbasis data historis dan "
        "bukan keputusan final institusi."
    )

    # ========================================================
    # FORM
    # ========================================================

    with st.form("prediction_form"):

        # ----------------------------------------------------
        # DATA DEMOGRAFIS
        # ----------------------------------------------------

        st.subheader("👤 Data Demografis")

        col1, col2, col3 = st.columns(3)

        with col1:
            marital_status = st.selectbox(
                "Marital Status",
                options=list(MARITAL_MAP.keys()),
                format_func=lambda x: MARITAL_MAP[x],
                index=0
            )

        with col2:
            gender = st.selectbox(
                "Gender",
                options=list(GENDER_MAP.keys()),
                format_func=lambda x: GENDER_MAP[x],
                index=1
            )

        with col3:
            age = st.number_input(
                "Age at Enrollment",
                min_value=15,
                max_value=80,
                value=20,
                step=1
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            displaced = st.selectbox(
                "Displaced",
                options=list(BINARY_MAP.keys()),
                format_func=lambda x: BINARY_MAP[x],
                index=0
            )

        with col2:
            special_needs = st.selectbox(
                "Educational Special Needs",
                options=list(BINARY_MAP.keys()),
                format_func=lambda x: BINARY_MAP[x],
                index=0
            )

        with col3:
            international = st.selectbox(
                "International",
                options=list(BINARY_MAP.keys()),
                format_func=lambda x: BINARY_MAP[x],
                index=0
            )

        # ----------------------------------------------------
        # DATA PENDIDIKAN
        # ----------------------------------------------------

        st.subheader("📚 Data Pendidikan")

        col1, col2, col3 = st.columns(3)

        with col1:
            application_mode = st.number_input(
                "Application Mode",
                min_value=1,
                max_value=60,
                value=1,
                step=1
            )

        with col2:
            application_order = st.number_input(
                "Application Order",
                min_value=0,
                max_value=10,
                value=1,
                step=1
            )

        with col3:
            course = st.number_input(
                "Course Code",
                min_value=1,
                max_value=100,
                value=1,
                step=1
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            daytime = st.selectbox(
                "Attendance",
                options=list(ATTENDANCE_MAP.keys()),
                format_func=lambda x: ATTENDANCE_MAP[x],
                index=1
            )

        with col2:
            previous_qualification = st.selectbox(
                "Previous Qualification",
                options=list(PREVIOUS_MAP.keys()),
                format_func=lambda x: PREVIOUS_MAP[x],
                index=0
            )

        with col3:
            previous_qualification_grade = st.number_input(
                "Previous Qualification Grade",
                min_value=0.0,
                max_value=200.0,
                value=120.0,
                step=0.1
            )

        # ----------------------------------------------------
        # NILAI DAN ADMISSION
        # ----------------------------------------------------

        st.subheader("🎯 Nilai Akademik dan Admission")

        col1, col2 = st.columns(2)

        with col1:
            admission_grade = st.number_input(
                "Admission Grade",
                min_value=0.0,
                max_value=200.0,
                value=120.0,
                step=0.1
            )

        with col2:
            tuition = st.selectbox(
                "Tuition Fees Up To Date",
                options=list(BINARY_MAP.keys()),
                format_func=lambda x: BINARY_MAP[x],
                index=1
            )

        # ----------------------------------------------------
        # KEUANGAN
        # ----------------------------------------------------

        st.subheader("💰 Kondisi Keuangan")

        col1, col2, col3 = st.columns(3)

        with col1:
            debtor = st.selectbox(
                "Debtor",
                options=list(BINARY_MAP.keys()),
                format_func=lambda x: BINARY_MAP[x],
                index=0
            )

        with col2:
            scholarship = st.selectbox(
                "Scholarship Holder",
                options=list(BINARY_MAP.keys()),
                format_func=lambda x: BINARY_MAP[x],
                index=0
            )

        with col3:
            nationality = st.number_input(
                "Nacionality Code",
                min_value=1,
                max_value=110,
                value=1,
                step=1
            )

        # ----------------------------------------------------
        # ORANG TUA
        # ----------------------------------------------------

        st.subheader("👨‍👩‍👦 Informasi Orang Tua")

        col1, col2 = st.columns(2)

        with col1:
            mothers_qualification = st.number_input(
                "Mother's Qualification",
                min_value=0,
                max_value=50,
                value=1,
                step=1
            )

        with col2:
            fathers_qualification = st.number_input(
                "Father's Qualification",
                min_value=0,
                max_value=50,
                value=1,
                step=1
            )

        col1, col2 = st.columns(2)

        with col1:
            mothers_occupation = st.number_input(
                "Mother's Occupation",
                min_value=0,
                max_value=200,
                value=1,
                step=1
            )

        with col2:
            fathers_occupation = st.number_input(
                "Father's Occupation",
                min_value=0,
                max_value=200,
                value=1,
                step=1
            )

        # ----------------------------------------------------
        # SEMESTER 1
        # ----------------------------------------------------

        st.subheader("📖 Performa Semester 1")

        col1, col2, col3 = st.columns(3)

        with col1:
            sem1_credited = st.number_input(
                "Semester 1 - Credited",
                min_value=0,
                value=0,
                step=1
            )

        with col2:
            sem1_enrolled = st.number_input(
                "Semester 1 - Jumlah Mata Kuliah Diambil",
                min_value=0,
                value=6,
                step=1
            )

        with col3:
            sem1_evaluations = st.number_input(
                "Semester 1 - Evaluations",
                min_value=0,
                value=6,
                step=1
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            sem1_approved = st.number_input(
                "Semester 1 - Approved",
                min_value=0,
                value=5,
                step=1
            )

        with col2:
            sem1_grade = st.number_input(
                "Semester 1 - Grade",
                min_value=0.0,
                max_value=20.0,
                value=12.0,
                step=0.1
            )

        with col3:
            sem1_without_eval = st.number_input(
                "Semester 1 - Without Evaluations",
                min_value=0,
                value=0,
                step=1
            )

        # ----------------------------------------------------
        # SEMESTER 2
        # ----------------------------------------------------

        st.subheader("📕 Performa Semester 2")

        col1, col2, col3 = st.columns(3)

        with col1:
            sem2_credited = st.number_input(
                "Semester 2 - Credited",
                min_value=0,
                value=0,
                step=1
            )

        with col2:
            sem2_enrolled = st.number_input(
                "Semester 2 - Jumlah Mata Kuliah Diambil",
                min_value=0,
                value=6,
                step=1
            )

        with col3:
            sem2_evaluations = st.number_input(
                "Semester 2 - Evaluations",
                min_value=0,
                value=6,
                step=1
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            sem2_approved = st.number_input(
                "Semester 2 - Approved",
                min_value=0,
                value=5,
                step=1
            )

        with col2:
            sem2_grade = st.number_input(
                "Semester 2 - Grade",
                min_value=0.0,
                max_value=20.0,
                value=12.0,
                step=0.1
            )

        with col3:
            sem2_without_eval = st.number_input(
                "Semester 2 - Without Evaluations",
                min_value=0,
                value=0,
                step=1
            )

        # ----------------------------------------------------
        # KONDISI EKONOMI
        # ----------------------------------------------------

        st.subheader("🌍 Kondisi Ekonomi")

        col1, col2, col3 = st.columns(3)

        with col1:
            unemployment = st.number_input(
                "Unemployment Rate",
                value=10.0,
                step=0.1
            )

        with col2:
            inflation = st.number_input(
                "Inflation Rate",
                value=2.0,
                step=0.1
            )

        with col3:
            gdp = st.number_input(
                "GDP",
                value=1.0,
                step=0.1
            )

        submitted = st.form_submit_button(
            "🔮 Prediksi Status Mahasiswa",
            type="primary",
            use_container_width=True
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    if submitted:

        input_data = pd.DataFrame([{
            "Marital_status": marital_status,
            "Application_mode": application_mode,
            "Application_order": application_order,
            "Course": course,
            "Daytime_evening_attendance": daytime,
            "Previous_qualification": previous_qualification,
            "Previous_qualification_grade": previous_qualification_grade,
            "Nacionality": nationality,
            "Mothers_qualification": mothers_qualification,
            "Fathers_qualification": fathers_qualification,
            "Mothers_occupation": mothers_occupation,
            "Fathers_occupation": fathers_occupation,
            "Admission_grade": admission_grade,
            "Displaced": displaced,
            "Educational_special_needs": special_needs,
            "Debtor": debtor,
            "Tuition_fees_up_to_date": tuition,
            "Gender": gender,
            "Scholarship_holder": scholarship,
            "Age_at_enrollment": age,
            "International": international,
            "Curricular_units_1st_sem_credited": sem1_credited,
            "Curricular_units_1st_sem_enrolled": sem1_enrolled,
            "Curricular_units_1st_sem_evaluations": sem1_evaluations,
            "Curricular_units_1st_sem_approved": sem1_approved,
            "Curricular_units_1st_sem_grade": sem1_grade,
            "Curricular_units_1st_sem_without_evaluations": sem1_without_eval,
            "Curricular_units_2nd_sem_credited": sem2_credited,
            "Curricular_units_2nd_sem_enrolled": sem2_enrolled,
            "Curricular_units_2nd_sem_evaluations": sem2_evaluations,
            "Curricular_units_2nd_sem_approved": sem2_approved,
            "Curricular_units_2nd_sem_grade": sem2_grade,
            "Curricular_units_2nd_sem_without_evaluations": sem2_without_eval,
            "Unemployment_rate": unemployment,
            "Inflation_rate": inflation,
            "GDP": gdp
        }])

        try:
            prediction = int(model.predict(input_data)[0])
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

        except Exception as e:
            st.error(
                "Prediksi gagal dijalankan. Pastikan model .pkl yang digunakan "
                "adalah model binary terbaru dari notebook revisi."
            )
            st.exception(e)
            st.stop()

        # ----------------------------------------------------
        # HASIL
        # ----------------------------------------------------

        st.divider()
        st.subheader("📌 Hasil Prediksi")

        # Mapping target binary sesuai notebook revisi:
        # 0 = Graduate, 1 = Dropout
        prediction_label = {
            0: "Graduate",
            1: "Dropout"
        }.get(prediction, str(prediction))

        st.info(
            "**Keterangan target model:** 0 = Graduate | 1 = Dropout"
        )

        if prediction == 1:

            st.error(
                "⚠️ Prediksi Status: **Dropout**"
            )

            st.warning(
                """
                Berdasarkan pola data yang dimasukkan, model memprediksi
                mahasiswa lebih dekat dengan kelompok **Dropout**.

                Institusi dapat mempertimbangkan monitoring dan intervensi
                lebih lanjut sesuai kondisi mahasiswa.
                """
            )

        else:

            st.success(
                "🎓 Prediksi Status: **Graduate**"
            )

            st.info(
                """
                Berdasarkan pola data yang dimasukkan, model memprediksi
                mahasiswa lebih dekat dengan kelompok **Graduate**.
                """
            )

        # ----------------------------------------------------
        # PROBABILITAS
        # ----------------------------------------------------

        st.subheader("📊 Probabilitas Prediksi")

        probability_labels = {
            0: "Graduate",
            1: "Dropout"
        }

        # Konversi kelas numerik model menjadi label yang mudah dipahami.
        # 0 = Graduate, 1 = Dropout.
        probability_df = pd.DataFrame({
            "Status Mahasiswa": [
                probability_labels.get(int(cls), "Tidak diketahui")
                for cls in classes
            ],
            "Probabilitas (%)": probabilities * 100
        })

        # Urutkan agar Graduate tampil sebelum Dropout.
        probability_df["Status Mahasiswa"] = pd.Categorical(
            probability_df["Status Mahasiswa"],
            categories=["Graduate", "Dropout"],
            ordered=True
        )
        probability_df = probability_df.sort_values("Status Mahasiswa")

        probability_df["Probabilitas (%)"] = (
            probability_df["Probabilitas (%)"].round(2)
        )

        # Tampilkan hanya label status, tanpa kode 0/1, agar hasil mudah
        # dipahami oleh pengguna non-teknis.
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

        with col2:
            chart_df = probability_df.set_index("Status Mahasiswa")
            st.bar_chart(
                chart_df["Probabilitas (%)"]
            )

        st.caption(
            "Probabilitas menunjukkan tingkat keyakinan model terhadap "
            "masing-masing kelas. Hasil prediksi bukan keputusan final "
            "dan tetap perlu dikombinasikan dengan evaluasi pihak institusi."
        )
