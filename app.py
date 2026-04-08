"""
🧠 Intelligent Data Analyzer

FLOW:
1. Upload Dataset
2. Analyze structure
3. Decision engine (checklist + reason)
4. Enable valid analysis only
5. Execute selected analysis
6. ML: auto-detect + evaluate
"""

import streamlit as st

# ✅ MUST BE FIRST
st.set_page_config(page_title="Intelligent Data Analyzer", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix
)

# =========================
# 🎨 THEME
# =========================
def apply_theme(theme):
    colors = {
        "Light": "#ffffff",
        "Dark": "#0f172a",
        "Blue": "#eff6ff",
        "Green": "#f0fdf4"
    }
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {colors[theme]}; }}
    </style>
    """, unsafe_allow_html=True)

# =========================
# 🔍 ANALYZER
# =========================
def detect_time_columns(df):
    cols = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        sample = df[col].dropna().astype(str).head(100)
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        if len(sample) > 0 and parsed.notna().mean() > 0.7:
            cols.append(col)
    return cols

def analyze_dataset(df):
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    time_cols = detect_time_columns(df)
    text_cols = [c for c in categorical if df[c].astype(str).str.len().mean() > 20]

    missing = df.isna().sum()

    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "numerical_cols": numerical,
        "categorical_cols": categorical,
        "time_cols": time_cols,
        "text_cols": text_cols,
        "clean_columns": int((missing == 0).sum()),
        "unclean_columns": int((missing > 0).sum()),
        "missing_per_column": missing.to_dict()
    }

def auto_feature_selection(df, target):
    drop_cols = []
    reasons = {}

    for col in df.columns:
        if col == target:
            continue

        # 1. High uniqueness (ID-like)
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.95:
            drop_cols.append(col)
            reasons[col] = "High uniqueness (ID-like)"
            continue

        # 2. Constant column
        if df[col].nunique() <= 1:
            drop_cols.append(col)
            reasons[col] = "Constant column"
            continue

        # 3. Too many missing values
        if df[col].isna().mean() > 0.5:
            drop_cols.append(col)
            reasons[col] = "Too many missing values"
            continue

        # 4. LOW CORRELATION (IMPORTANT NEW)
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target]):
            corr_matrix = df[[col, target]].corr()

            if corr_matrix.shape[0] > 1:
                corr = corr_matrix.iloc[0, 1]

                if pd.notna(corr) and abs(corr) < 0.01:
                    drop_cols.append(col)
                    reasons[col] = "Low correlation with target"
                    continue


    return drop_cols, reasons

# =========================
# 🧠 DECISION ENGINE
# =========================
def decide_analysis(f):
    return [
        {"name":"Statistical","enabled":len(f["numerical_cols"])>0,
         "checklist":{"Has numerical columns":len(f["numerical_cols"])>0}},
        {"name":"EDA","enabled":len(f["numerical_cols"])>0 and len(f["categorical_cols"])>0,
         "checklist":{"Has numerical columns":len(f["numerical_cols"])>0,
                      "Has categorical columns":len(f["categorical_cols"])>0}},
        {"name":"Time Series","enabled":len(f["time_cols"])>0,
         "checklist":{"Has time column":len(f["time_cols"])>0}},
        {"name":"Text Analysis","enabled":len(f["text_cols"])>0,
         "checklist":{"Has text column":len(f["text_cols"])>0}},
        {"name":"Data Cleaning","enabled":f["unclean_columns"]>0,
         "checklist":{"Has missing values":f["unclean_columns"]>0}}
    ]

# =========================
# 🤖 ML ENGINE
# =========================
def decide_ml(df, target):
    y = df[target].dropna()

    if len(y) == 0:
        return None, "No valid target", []

    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 10:
            return (
                "Classification",
                "Few unique numeric values",
                ["Logistic Regression", "Decision Tree", "Random Forest"]
            )
        else:
            return (
                "Regression",
                "Continuous numeric target",
                ["Linear Regression", "Decision Tree", "Random Forest"]
            )

    return (
        "Classification",
        "Categorical target",
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )


# =========================
# 🎛️ SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")

theme = st.sidebar.selectbox("Theme",["Light","Dark","Blue","Green"])
apply_theme(theme)

file = st.sidebar.file_uploader("Upload CSV",type=["csv"])

# =========================
# 🚀 MAIN
# =========================
st.title("🧠 Intelligent Data Analyzer")

if file:
    df = pd.read_csv(file)

    target = st.sidebar.selectbox("Select Target Column", df.columns)

    f = analyze_dataset(df)

    # KPI
    st.subheader("📊 Dataset Overview")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Rows",f["n_rows"])
    c2.metric("Columns",f["n_cols"])
    c3.metric("Numerical",len(f["numerical_cols"]))
    c4.metric("Categorical",len(f["categorical_cols"]))
    c5.metric("Clean",f["clean_columns"])
    c6.metric("Missing",f["unclean_columns"])

    st.dataframe(df.head())

    # ================= DECISION =================
    st.subheader("🧠 Decision Engine")
    decisions = decide_analysis(f)
    enabled = []

    for d in decisions:
        if d["enabled"]:
            st.success(f"✅ {d['name']}")
            enabled.append(d["name"])
        else:
            st.error(f"❌ {d['name']}")

        st.write("Checklist:")
        for k,v in d["checklist"].items():
            st.write(f"{'✔' if v else '✖'} {k}")

        reason = ", ".join([k for k,v in d["checklist"].items() if v])
        st.info(f"Reason: {reason if reason else 'Conditions not satisfied'}")

        st.markdown("---")

    # ================= ML =================
    st.subheader("🤖 Machine Learning")
    ml_type, reason, models = decide_ml(df, target)

    if ml_type:
        st.success(f"{ml_type}")
        st.info(reason)

        st.write("Suggested Models:")
        for m in models:
            st.write(f"✔ {m}")

        enabled.append("Machine Learning")

    choice = st.selectbox("Select Analysis", enabled)

    # ================= EXECUTION =================

    if choice=="Statistical":
        st.write(df.describe())


    elif choice == "EDA":

        st.subheader("📊 EDA Visualizations")

        num_cols = df.select_dtypes(include=["number"]).columns

        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        # ================= DISTRIBUTION =================

        st.write("🔹 Distribution of Numerical Columns")

        for col in num_cols:
            fig, ax = plt.subplots()

            sns.histplot(df[col], kde=True, ax=ax)

            ax.set_title(f"Distribution of {col}")

            st.pyplot(fig)

        # ================= BOXPLOT =================

        st.write("🔹 Boxplots (Outliers Detection)")

        for col in num_cols:
            fig, ax = plt.subplots()

            sns.boxplot(x=df[col], ax=ax)

            ax.set_title(f"Boxplot of {col}")

            st.pyplot(fig)

        # ================= CORRELATION =================

        if len(num_cols) >= 2:
            st.write("🔹 Correlation Heatmap")

            fig, ax = plt.subplots()

            sns.heatmap(df[num_cols].corr(), annot=True, cmap="Blues", ax=ax)

            st.pyplot(fig)

        # ================= CATEGORY ANALYSIS =================

        if len(cat_cols) > 0 and len(num_cols) > 0:
            st.write("🔹 Category vs Numerical")

            cat = st.selectbox("Select Category", cat_cols)

            num = st.selectbox("Select Numeric", num_cols)

            fig, ax = plt.subplots()

            sns.barplot(x=cat, y=num, data=df, ax=ax)

            plt.xticks(rotation=45)

            st.pyplot(fig)

    elif choice=="Time Series":
        if f["time_cols"] and f["numerical_cols"]:
            t = st.selectbox("Time",f["time_cols"])
            n = st.selectbox("Value",f["numerical_cols"])

            temp = df.copy()
            temp[t] = pd.to_datetime(temp[t],errors="coerce")
            temp = temp.dropna(subset=[t])
            temp = temp.sort_values(t)

            fig,ax = plt.subplots()
            ax.plot(temp[t],temp[n])
            st.pyplot(fig)

    elif choice=="Text Analysis":
        if f["text_cols"]:
            col = st.selectbox("Text Column",f["text_cols"])
            text = df[col].dropna().astype(str)

            lengths = text.apply(len)

            fig,ax = plt.subplots()
            ax.hist(lengths,bins=20)
            st.pyplot(fig)

    elif choice=="Data Cleaning":
        st.write(f["missing_per_column"])

    elif choice=="Machine Learning":

        data = df.dropna(subset=[target]).copy()
        # ================= AUTO FEATURE SELECTION =================
        drop_cols, reasons = auto_feature_selection(data, target)

        st.subheader("🧹 Auto Feature Selection")

        if drop_cols:
            st.write("Dropped Columns:")
            for col in drop_cols:
                st.write(f"❌ {col} → {reasons[col]}")
        else:
            st.write("No columns dropped")

        # Apply drop
        clean_data = data.drop(columns=drop_cols)

        if clean_data.shape[1] <= 1:
            st.error("All features removed after cleaning. Please check dataset.")
            st.stop()

        # Prepare features
        X = pd.get_dummies(clean_data.drop(columns=[target]), drop_first=True)
        y = clean_data[target]

        st.write("✅ Final Features Used:", list(X.columns))


        if not pd.api.types.is_numeric_dtype(y):
            y = y.astype(str)

        if y.nunique()<2:
            st.error("Not enough classes")
            st.stop()

        if X.shape[1]==0:
            st.error("No usable features")
            st.stop()

        # Mean works because all features numeric after encoding
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X),columns=X.columns)

        strat = y if ml_type=="Classification" and min(y.value_counts())>=2 else None

        X_train,X_test,y_train,y_test = train_test_split(
            X,y,test_size=0.2,random_state=42,stratify=strat
        )

        if ml_type == "Regression":

            model = LinearRegression()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            st.write("MAE:", mean_absolute_error(y_test, pred))
            st.write("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
            st.write("R2 Score:", r2_score(y_test, pred))

            st.subheader("📈 Regression Visualizations")

            # Actual vs Predicted
            fig, ax = plt.subplots()
            ax.scatter(y_test, pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

            # Residuals
            residuals = y_test - pred
            fig, ax = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_title("Residual Distribution")
            st.pyplot(fig)



        else:

            model = LogisticRegression(max_iter=1000)

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            st.subheader("📊 Classification Visualizations")
            st.write("Accuracy:", accuracy_score(y_test, pred))

            cm = confusion_matrix(y_test, pred)

            fig, ax = plt.subplots()

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

            ax.set_title("Confusion Matrix")

            st.pyplot(fig)

else:
    st.info("Upload dataset to begin 🚀")