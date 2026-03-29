import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Student Performance Analyzer", page_icon="🎓", layout="centered")

SUBJECTS = ["EMWTL", "EDC", "Signals and Systems", "Python", "Analog Circuits"]

@st.cache_resource
def train_models():
    np.random.seed(42)
    df = pd.DataFrame({s: np.random.randint(20, 100, 72) for s in SUBJECTS})
    df["Attendance"] = np.random.randint(50, 100, 72)
    df["Backlogs"] = sum((df[s] < 35).astype(int) for s in SUBJECTS)
    df["CGPA"] = np.where(df["Backlogs"] == 0, df[SUBJECTS].sum(axis=1) / 50, 0)
    df["Placed"] = ((df["CGPA"] > 7) & (df["Backlogs"] == 0) & (df["Attendance"] > 75)).astype(int)
    df["Fail"] = (df["Backlogs"] > 0).astype(int)

    scale_cols = SUBJECTS + ["Attendance", "CGPA"]
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    def fit(X, y):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression() if y.name == "Fail" else RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(Xtr, ytr)
        return model, round(accuracy_score(yte, model.predict(Xte)), 2)

    features = df.drop(["Placed", "Fail"], axis=1)
    fail_model, fail_acc = fit(features, df["Fail"])
    place_model, place_acc = fit(features, df["Placed"])
    return fail_model, place_model, scaler, fail_acc, place_acc

@st.cache_data
def load_student_data():
    return pd.read_csv("student_data.csv")

fail_model, place_model, scaler, fail_acc, place_acc = train_models()

try:
    student_df = load_student_data()
    if "Name" not in student_df.columns:
        st.error("Column 'Name' not found in student_data.csv")
        st.stop()
except Exception as e:
    st.error(f"Error loading student_data.csv: {e}")
    st.stop()

student_names = student_df["Name"].dropna().astype(str).tolist()

st.sidebar.title("Model Accuracy")
st.sidebar.metric("Failure Model", f"{fail_acc*100:.0f}%")
st.sidebar.metric("Placement Model", f"{place_acc*100:.0f}%")
st.sidebar.caption("Trained on 72 synthetic records.")

st.title("AI Student Performance Analyzer")
st.caption("Select a student, auto-fill marks, edit if needed, and analyze performance.")
st.divider()

name = st.selectbox(
    "Student Name",
    options=student_names,
    index=None,
    placeholder="Choose or search a student"
)

default_vals = [60, 60, 60, 60, 60]
default_attendance = 80

if name:
    selected_row = student_df[student_df["Name"] == name].iloc[0]
    default_vals = [int(selected_row[sub]) for sub in SUBJECTS]
    default_attendance = int(selected_row["Attendance"])

st.subheader("Marks & Attendance")

c1, c2 = st.columns(2)
vals = [
    c1.slider("EMWTL", 0, 100, default_vals[0]),
    c2.slider("EDC", 0, 100, default_vals[1]),
    c1.slider("Signals and Systems", 0, 100, default_vals[2]),
    c2.slider("Python", 0, 100, default_vals[3]),
    c1.slider("Analog Circuits", 0, 100, default_vals[4]),
]
attendance = c2.slider("Attendance (%)", 50, 100, default_attendance)

st.divider()

if st.button("Analyze Student", type="primary", use_container_width=True):
    marks = dict(zip(SUBJECTS, vals))
    backlog = sum(1 for v in vals if v < 35)
    cgpa = round(sum(vals) / 50, 2) if backlog == 0 else 0
    avg = sum(vals) / len(vals)

    input_df = pd.DataFrame(
        [vals + [attendance, backlog, cgpa]],
        columns=SUBJECTS + ["Attendance", "Backlogs", "CGPA"]
    )
    input_df[SUBJECTS + ["Attendance", "CGPA"]] = scaler.transform(
        input_df[SUBJECTS + ["Attendance", "CGPA"]]
    )

    fail = fail_model.predict(input_df)[0]
    place = place_model.predict(input_df)[0]
    weak = [s for s, v in marks.items() if v < 60]

    st.subheader(f"Results — {name or 'Student'}")

    m1, m2, m3 = st.columns(3)
    m1.metric("CGPA", cgpa)
    m2.metric("Backlogs", backlog)
    m3.metric("Attendance", f"{attendance}%")

    st.divider()

    r1, r2 = st.columns(2)
    with r1:
        st.subheader("Failure Risk")
        if fail:
            st.error("High — Fix backlogs immediately.")
        else:
            st.success("Low — No backlogs.")
    with r2:
        st.subheader("Placement Chance")
        if place:
            st.success("High — Meets placement criteria.")
        else:
            st.warning("Low — Improve CGPA / attendance.")

    st.divider()

    st.subheader("Subjects to Focus On")
    if weak:
        for sub in weak:
            tag = f"CRITICAL — {sub} ({marks[sub]}/100)" if marks[sub] < 35 else f"NEEDS WORK — {sub} ({marks[sub]}/100)"
            with st.expander(tag):
                st.write("Failing — prioritise immediately." if marks[sub] < 35 else "Below 60 — needs regular practice.")
    else:
        st.success("All subjects above 60!")

    st.divider()

    st.subheader("Counselor Advice")
    n = name.split()[0] if name else "You"
    failing = [s for s, v in marks.items() if v < 35]
    advice = (
        f"{n}, you're failing {', '.join(failing)}. Focus here first." if failing else
        "You're in good shape. Don't get complacent — tighten up the weaker ones." if avg >= 80 else
        "Not bad, but close the gap between your best and worst subjects." if avg >= 65 else
        "Passing, but barely. Consistent daily effort — not just before exams." if avg >= 50 else
        "One focused semester can turn this around. Start with your worst subject today."
    )
    st.info(advice)

    st.divider()

    ch1, ch2 = st.columns(2)

    with ch1:
        st.subheader("Performance Chart")
        colors = ["#ef4444" if v < 35 else "#facc15" if v < 60 else "#22c55e" for v in vals]
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        fig1.patch.set_facecolor("#0f172a")
        ax1.set_facecolor("#1e293b")
        bars = ax1.bar(SUBJECTS, vals, color=colors, width=0.5)
        ax1.axhline(35, color="#ef4444", linestyle="--", linewidth=1, label="Pass (35)")
        ax1.axhline(60, color="#facc15", linestyle="--", linewidth=1, label="Good (60)")
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, v + 1.5, str(v), ha="center", fontsize=9, color="white")
        ax1.set_ylim(0, 110)
        ax1.set_ylabel("Marks", color="#94a3b8")
        ax1.set_title("Subject Marks", color="#f1f5f9", pad=8)
        ax1.tick_params(colors="#94a3b8")
        ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax1.legend(fontsize=7, facecolor="#1e293b", labelcolor="#94a3b8")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig1)

    with ch2:
        st.subheader("Placement Chance")
        place_pct = min(100, max(5,
            (cgpa / 10) * 40 +
            (attendance / 100) * 30 +
            (0 if backlog > 0 else 20) +
            (avg / 100) * 10
        ))
        not_place_pct = 100 - place_pct
        pie_color = "#22c55e" if place_pct >= 70 else "#facc15" if place_pct >= 45 else "#ef4444"

        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        fig2.patch.set_facecolor("#0f172a")
        ax2.set_facecolor("#0f172a")
        ax2.pie(
            [place_pct, not_place_pct],
            colors=[pie_color, "#1e293b"],
            startangle=90,
            wedgeprops={"width": 0.45, "edgecolor": "#b3bed5", "linewidth": 2}
        )
        ax2.text(0, 0, f"{place_pct:.0f}%", ha="center", va="center",
                 fontsize=26, fontweight="bold", color=pie_color)
        ax2.set_title("Placement Chance", color="#f1f5f9", pad=8)
        plt.tight_layout()
        st.pyplot(fig2)

        label = "High 🟢" if place_pct >= 70 else "Moderate 🟡" if place_pct >= 45 else "Low 🔴"
        st.caption(f"Placement likelihood: **{label}**")