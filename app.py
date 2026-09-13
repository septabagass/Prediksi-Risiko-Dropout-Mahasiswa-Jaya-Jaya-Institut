import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt


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

MODEL_PATH = "model/dropout_prediction_pipeline.pkl"
DATA_PATH = "data/dashboard_data.csv"


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
        "File model tidak ditemukan. "
        "Pastikan file berada di folder model/."
    )
    st.stop()


if not os.path.exists(DATA_PATH):
    st.error(
        "File dashboard_data.csv tidak ditemukan. "
        "Pastikan file berada di folder data/."
    )
    st.stop()


model = load_model()
df = load_data()


# ============================================================
# HEADER
# ============================================================

st.title("🎓 Jaya Jaya Institut")
st.subheader("Sistem Monitoring dan Prediksi Risiko Dropout Mahasiswa")

st.write(
    """
    Aplikasi ini membantu pihak Jaya Jaya Institut dalam memahami
    kondisi mahasiswa berdasarkan data akademik, demografis,
    dan kondisi pendukung lainnya.

    Aplikasi terdiri dari dua bagian utama:
    - 📊 **Dashboard** untuk memantau kondisi mahasiswa.
    - 🤖 **Prediksi Risiko Dropout** untuk memprediksi status mahasiswa.
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
        Dashboard berikut memberikan gambaran mengenai distribusi
        status mahasiswa dan beberapa faktor yang berkaitan dengan
        performa akademik serta status mahasiswa.
        """
    )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    st.sidebar.divider()
    st.sidebar.subheader("🔎 Filter Dashboard")

    status_options = sorted(
        df["Status"].dropna().unique().tolist()
    )

    selected_status = st.sidebar.multiselect(
        "Status Mahasiswa",
        options=status_options,
        default=status_options
    )

    filtered_df = df[
        df["Status"].isin(selected_status)
    ].copy()

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    total_students = len(filtered_df)

    dropout_students = len(
        filtered_df[
            filtered_df["Status"] == "Dropout"
        ]
    )

    graduate_students = len(
        filtered_df[
            filtered_df["Status"] == "Graduate"
        ]
    )

    enrolled_students = len(
        filtered_df[
            filtered_df["Status"] == "Enrolled"
        ]
    )

    if total_students > 0:
        dropout_rate = (
            dropout_students / total_students
        ) * 100
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

    st.divider()

    # --------------------------------------------------------
    # STATUS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("1. Distribusi Status Mahasiswa")

    status_count = (
        filtered_df["Status"]
        .value_counts()
        .reindex(
            ["Dropout", "Enrolled", "Graduate"],
            fill_value=0
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        status_count.index,
        status_count.values
    )

    ax.set_title(
        "Distribusi Status Mahasiswa"
    )

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

    # --------------------------------------------------------
    # COURSE
    # --------------------------------------------------------

    st.subheader("2. Status Mahasiswa Berdasarkan Program Studi")

    course_status = pd.crosstab(
        filtered_df["Course"],
        filtered_df["Status"]
    )

    for status in [
        "Dropout",
        "Enrolled",
        "Graduate"
    ]:
        if status not in course_status.columns:
            course_status[status] = 0

    course_status = course_status[
        ["Dropout", "Enrolled", "Graduate"]
    ]

    course_status = course_status.sort_values(
        "Dropout",
        ascending=False
    ).head(10)

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    course_status.plot(
        kind="bar",
        stacked=False,
        ax=ax
    )

    ax.set_title(
        "Top 10 Program Studi Berdasarkan Jumlah Dropout"
    )

    ax.set_xlabel("Course")
    ax.set_ylabel("Jumlah Mahasiswa")

    ax.tick_params(axis="x", rotation=45)

    ax.legend(
        title="Status"
    )

    st.pyplot(fig)

    st.caption(
        "Program studi diurutkan berdasarkan jumlah mahasiswa dropout."
    )

    # --------------------------------------------------------
    # TUITION
    # --------------------------------------------------------

    st.subheader(
        "3. Status Mahasiswa Berdasarkan Kondisi Pembayaran"
    )

    if "Tuition_fees_up_to_date_label" in filtered_df.columns:

        tuition_status = pd.crosstab(
            filtered_df[
                "Tuition_fees_up_to_date_label"
            ],
            filtered_df["Status"]
        )

        for status in [
            "Dropout",
            "Enrolled",
            "Graduate"
        ]:
            if status not in tuition_status.columns:
                tuition_status[status] = 0

        tuition_status = tuition_status[
            ["Dropout", "Enrolled", "Graduate"]
        ]

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        tuition_status.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Status Mahasiswa Berdasarkan Pembayaran Tuition Fees"
        )

        ax.set_xlabel(
            "Status Pembayaran"
        )

        ax.set_ylabel(
            "Jumlah Mahasiswa"
        )

        ax.tick_params(
            axis="x",
            rotation=0
        )

        ax.legend(
            title="Status Mahasiswa"
        )

        st.pyplot(fig)

    # --------------------------------------------------------
    # SCHOLARSHIP
    # --------------------------------------------------------

    st.subheader(
        "4. Status Mahasiswa Berdasarkan Beasiswa"
    )

    if "Scholarship_holder_label" in filtered_df.columns:

        scholarship_status = pd.crosstab(
            filtered_df[
                "Scholarship_holder_label"
            ],
            filtered_df["Status"]
        )

        for status in [
            "Dropout",
            "Enrolled",
            "Graduate"
        ]:
            if status not in scholarship_status.columns:
                scholarship_status[status] = 0

        scholarship_status = scholarship_status[
            ["Dropout", "Enrolled", "Graduate"]
        ]

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        scholarship_status.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Status Mahasiswa Berdasarkan Kepemilikan Beasiswa"
        )

        ax.set_xlabel(
            "Status Beasiswa"
        )

        ax.set_ylabel(
            "Jumlah Mahasiswa"
        )

        ax.tick_params(
            axis="x",
            rotation=0
        )

        ax.legend(
            title="Status Mahasiswa"
        )

        st.pyplot(fig)

    # --------------------------------------------------------
    # SEMESTER 1
    # --------------------------------------------------------

    st.subheader(
        "5. Performa Akademik Semester 1"
    )

    semester1_summary = (
        filtered_df
        .groupby("Status")[
            [
                "Curricular_units_1st_sem_approved",
                "Curricular_units_1st_sem_grade"
            ]
        ]
        .mean()
        .reindex(
            ["Dropout", "Enrolled", "Graduate"]
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        semester1_summary[
            "Curricular_units_1st_sem_approved"
        ].plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Rata-rata Mata Kuliah Lulus Semester 1"
        )

        ax.set_xlabel("Status")
        ax.set_ylabel("Rata-rata Mata Kuliah Lulus")

        ax.tick_params(
            axis="x",
            rotation=0
        )

        st.pyplot(fig)

    with col2:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        semester1_summary[
            "Curricular_units_1st_sem_grade"
        ].plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Rata-rata Nilai Semester 1"
        )

        ax.set_xlabel("Status")
        ax.set_ylabel("Rata-rata Nilai")

        ax.tick_params(
            axis="x",
            rotation=0
        )

        st.pyplot(fig)

    # --------------------------------------------------------
    # SEMESTER 2
    # --------------------------------------------------------

    st.subheader(
        "6. Performa Akademik Semester 2"
    )

    semester2_summary = (
        filtered_df
        .groupby("Status")[
            [
                "Curricular_units_2nd_sem_approved",
                "Curricular_units_2nd_sem_grade"
            ]
        ]
        .mean()
        .reindex(
            ["Dropout", "Enrolled", "Graduate"]
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        semester2_summary[
            "Curricular_units_2nd_sem_approved"
        ].plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Rata-rata Mata Kuliah Lulus Semester 2"
        )

        ax.set_xlabel("Status")
        ax.set_ylabel("Rata-rata Mata Kuliah Lulus")

        ax.tick_params(
            axis="x",
            rotation=0
        )

        st.pyplot(fig)

    with col2:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        semester2_summary[
            "Curricular_units_2nd_sem_grade"
        ].plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Rata-rata Nilai Semester 2"
        )

        ax.set_xlabel("Status")
        ax.set_ylabel("Rata-rata Nilai")

        ax.tick_params(
            axis="x",
            rotation=0
        )

        st.pyplot(fig)

    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------

    st.subheader(
        "7. Status Mahasiswa Berdasarkan Gender"
    )

    if "Gender_label" in filtered_df.columns:

        gender_status = pd.crosstab(
            filtered_df["Gender_label"],
            filtered_df["Status"]
        )

        for status in [
            "Dropout",
            "Enrolled",
            "Graduate"
        ]:
            if status not in gender_status.columns:
                gender_status[status] = 0

        gender_status = gender_status[
            ["Dropout", "Enrolled", "Graduate"]
        ]

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        gender_status.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Status Mahasiswa Berdasarkan Gender"
        )

        ax.set_xlabel("Gender")
        ax.set_ylabel("Jumlah Mahasiswa")

        ax.tick_params(
            axis="x",
            rotation=0
        )

        ax.legend(
            title="Status Mahasiswa"
        )

        st.pyplot(fig)

    # --------------------------------------------------------
    # INSIGHT
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "💡 Insight Utama"
    )

    st.markdown(
        """
        Berdasarkan dashboard, beberapa aspek yang perlu diperhatikan
        oleh pihak institusi antara lain:

        - **Status akademik mahasiswa** perlu dipantau sejak semester awal.
        - Jumlah mata kuliah yang berhasil diselesaikan dan nilai semester
          dapat digunakan sebagai indikator penting dalam monitoring mahasiswa.
        - Kondisi pembayaran tuition fees perlu diperhatikan sebagai salah
          satu aspek pendukung dalam pemantauan mahasiswa.
        - Kepemilikan beasiswa dapat digunakan sebagai salah satu informasi
          tambahan dalam memahami kondisi mahasiswa.
        - Perbedaan distribusi status antar program studi dapat menjadi dasar
          untuk menentukan prioritas monitoring.
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

    st.header("🤖 Prediksi Status Mahasiswa")

    st.write(
        """
        Masukkan informasi mahasiswa pada form berikut.
        Model akan memprediksi status mahasiswa menjadi:

        **Dropout, Enrolled, atau Graduate.**
        """
    )

    st.divider()

    # --------------------------------------------------------
    # DATA DEMOGRAFIS
    # --------------------------------------------------------

    st.subheader("👤 Data Demografis")

    col1, col2, col3 = st.columns(3)

    with col1:

        marital_status = st.number_input(
            "Marital Status",
            min_value=1,
            max_value=6,
            value=1
        )

    with col2:

        gender = st.number_input(
            "Gender",
            min_value=0,
            max_value=1,
            value=1
        )

    with col3:

        age = st.number_input(
            "Age at Enrollment",
            min_value=15,
            max_value=80,
            value=20
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        displaced = st.number_input(
            "Displaced",
            min_value=0,
            max_value=1,
            value=0
        )

    with col2:

        special_needs = st.number_input(
            "Educational Special Needs",
            min_value=0,
            max_value=1,
            value=0
        )

    with col3:

        international = st.number_input(
            "International",
            min_value=0,
            max_value=1,
            value=0
        )

    # --------------------------------------------------------
    # DATA PENDIDIKAN
    # --------------------------------------------------------

    st.subheader("📚 Data Pendidikan")

    col1, col2, col3 = st.columns(3)

    with col1:

        application_mode = st.number_input(
            "Application Mode",
            min_value=1,
            max_value=60,
            value=1
        )

    with col2:

        application_order = st.number_input(
            "Application Order",
            min_value=0,
            max_value=10,
            value=1
        )

    with col3:

        course = st.number_input(
            "Course",
            min_value=1,
            max_value=100,
            value=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        daytime = st.number_input(
            "Daytime / Evening Attendance",
            min_value=0,
            max_value=1,
            value=1
        )

    with col2:

        previous_qualification = st.number_input(
            "Previous Qualification",
            min_value=1,
            max_value=50,
            value=1
        )

    with col3:

        previous_qualification_grade = st.number_input(
            "Previous Qualification Grade",
            min_value=0.0,
            max_value=200.0,
            value=120.0
        )

    # --------------------------------------------------------
    # NILAI
    # --------------------------------------------------------

    st.subheader("🎯 Nilai Akademik dan Admission")

    col1, col2 = st.columns(2)

    with col1:

        admission_grade = st.number_input(
            "Admission Grade",
            min_value=0.0,
            max_value=200.0,
            value=120.0
        )

    with col2:

        tuition = st.number_input(
            "Tuition Fees Up To Date",
            min_value=0,
            max_value=1,
            value=1
        )

    # --------------------------------------------------------
    # KEUANGAN
    # --------------------------------------------------------

    st.subheader("💰 Kondisi Keuangan")

    col1, col2, col3 = st.columns(3)

    with col1:

        debtor = st.number_input(
            "Debtor",
            min_value=0,
            max_value=1,
            value=0
        )

    with col2:

        scholarship = st.number_input(
            "Scholarship Holder",
            min_value=0,
            max_value=1,
            value=0
        )

    with col3:

        nationality = st.number_input(
            "Nacionality",
            min_value=1,
            max_value=50,
            value=1
        )

    # --------------------------------------------------------
    # ORANG TUA
    # --------------------------------------------------------

    st.subheader("👨‍👩‍👦 Informasi Orang Tua")

    col1, col2 = st.columns(2)

    with col1:

        mothers_qualification = st.number_input(
            "Mother's Qualification",
            min_value=0,
            max_value=50,
            value=1
        )

    with col2:

        fathers_qualification = st.number_input(
            "Father's Qualification",
            min_value=0,
            max_value=50,
            value=1
        )

    col1, col2 = st.columns(2)

    with col1:

        mothers_occupation = st.number_input(
            "Mother's Occupation",
            min_value=0,
            max_value=200,
            value=1
        )

    with col2:

        fathers_occupation = st.number_input(
            "Father's Occupation",
            min_value=0,
            max_value=200,
            value=1
        )

    # --------------------------------------------------------
    # SEMESTER 1
    # --------------------------------------------------------

    st.subheader("📖 Performa Semester 1")

    col1, col2, col3 = st.columns(3)

    with col1:

        sem1_credited = st.number_input(
            "Semester 1 - Credited",
            min_value=0,
            value=0
        )

    with col2:

        sem1_enrolled = st.number_input(
            "Semester 1 - Enrolled",
            min_value=0,
            value=6
        )

    with col3:

        sem1_evaluations = st.number_input(
            "Semester 1 - Evaluations",
            min_value=0,
            value=6
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        sem1_approved = st.number_input(
            "Semester 1 - Approved",
            min_value=0,
            value=5
        )

    with col2:

        sem1_grade = st.number_input(
            "Semester 1 - Grade",
            min_value=0.0,
            max_value=20.0,
            value=12.0
        )

    with col3:

        sem1_without_eval = st.number_input(
            "Semester 1 - Without Evaluations",
            min_value=0,
            value=0
        )

    # --------------------------------------------------------
    # SEMESTER 2
    # --------------------------------------------------------

    st.subheader("📕 Performa Semester 2")

    col1, col2, col3 = st.columns(3)

    with col1:

        sem2_credited = st.number_input(
            "Semester 2 - Credited",
            min_value=0,
            value=0
        )

    with col2:

        sem2_enrolled = st.number_input(
            "Semester 2 - Enrolled",
            min_value=0,
            value=6
        )

    with col3:

        sem2_evaluations = st.number_input(
            "Semester 2 - Evaluations",
            min_value=0,
            value=6
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        sem2_approved = st.number_input(
            "Semester 2 - Approved",
            min_value=0,
            value=5
        )

    with col2:

        sem2_grade = st.number_input(
            "Semester 2 - Grade",
            min_value=0.0,
            max_value=20.0,
            value=12.0
        )

    with col3:

        sem2_without_eval = st.number_input(
            "Semester 2 - Without Evaluations",
            min_value=0,
            value=0
        )

    # --------------------------------------------------------
    # KONDISI EKONOMI
    # --------------------------------------------------------

    st.subheader("🌍 Kondisi Ekonomi")

    col1, col2, col3 = st.columns(3)

    with col1:

        unemployment = st.number_input(
            "Unemployment Rate",
            value=10.0
        )

    with col2:

        inflation = st.number_input(
            "Inflation Rate",
            value=2.0
        )

    with col3:

        gdp = st.number_input(
            "GDP",
            value=1.0
        )

    # --------------------------------------------------------
    # PREDIKSI
    # --------------------------------------------------------

    st.divider()

    predict_button = st.button(
        "🔮 Prediksi Status Mahasiswa",
        type="primary",
        use_container_width=True
    )

    if predict_button:

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

            "Curricular_units_1st_sem_credited":
                sem1_credited,

            "Curricular_units_1st_sem_enrolled":
                sem1_enrolled,

            "Curricular_units_1st_sem_evaluations":
                sem1_evaluations,

            "Curricular_units_1st_sem_approved":
                sem1_approved,

            "Curricular_units_1st_sem_grade":
                sem1_grade,

            "Curricular_units_1st_sem_without_evaluations":
                sem1_without_eval,

            "Curricular_units_2nd_sem_credited":
                sem2_credited,

            "Curricular_units_2nd_sem_enrolled":
                sem2_enrolled,

            "Curricular_units_2nd_sem_evaluations":
                sem2_evaluations,

            "Curricular_units_2nd_sem_approved":
                sem2_approved,

            "Curricular_units_2nd_sem_grade":
                sem2_grade,

            "Curricular_units_2nd_sem_without_evaluations":
                sem2_without_eval,

            "Unemployment_rate":
                unemployment,

            "Inflation_rate":
                inflation,

            "GDP":
                gdp
        }])

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        classes = model.classes_

        # ----------------------------------------------------
        # HASIL
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📌 Hasil Prediksi"
        )

        if prediction == "Dropout":

            st.error(
                f"⚠️ Prediksi Status: **{prediction}**"
            )

            st.warning(
                """
                Mahasiswa teridentifikasi memiliki status
                **Dropout** berdasarkan pola data yang dimasukkan.

                Institusi dapat mempertimbangkan monitoring dan
                intervensi lebih lanjut.
                """
            )

        elif prediction == "Graduate":

            st.success(
                f"🎓 Prediksi Status: **{prediction}**"
            )

        else:

            st.info(
                f"📚 Prediksi Status: **{prediction}**"
            )

        # ----------------------------------------------------
        # PROBABILITAS
        # ----------------------------------------------------

        st.subheader(
            "📊 Probabilitas Prediksi"
        )

        probability_df = pd.DataFrame({

            "Status": classes,

            "Probability": probabilities * 100

        })

        probability_df[
            "Probability"
        ] = probability_df[
            "Probability"
        ].round(2)

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------

        st.bar_chart(
            probability_df.set_index(
                "Status"
            )["Probability"]
        )

        st.caption(
            "Probabilitas menunjukkan tingkat keyakinan model "
            "terhadap masing-masing status. Hasil prediksi bukan "
            "merupakan keputusan final dan tetap perlu dikombinasikan "
            "dengan evaluasi pihak institusi."
        )