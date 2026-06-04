from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="نظام ذكي للتنبؤ بمستويات ازدحام المعتمرين",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


MODEL_PATHS = [
    Path("models/growth_rate_xgboost_model.pkl"),
    Path("growth_rate_xgboost_model.pkl")
]

DATE_COL = "Gregorian_Date"
MONTH_COL = "Hijri_Month"
TARGET_COL = "المعتمرين"

HIJRI_MONTHS = [
    "محرم", "صفر", "ربيع الأول", "ربيع الآخر",
    "جماد الأول", "جماد الآخر",
    "رجب", "شعبان", "رمضان", "شوال",
    "ذو القعدة", "ذو الحجة"
]

NATIONALITY_OPTIONS = ["سعودي", "غير سعودي"]

LEVEL_COLORS = {
    "منخفض": "#1A9B6C",
    "متوسط": "#C8921E",
    "مرتفع": "#D05045"
}

LEVEL_BG = {
    "منخفض": "#EEF9F3",
    "متوسط": "#FFF8E8",
    "مرتفع": "#FFF1F0"
}

LEVEL_BORDER = {
    "منخفض": "#A8D9BF",
    "متوسط": "#E8CC94",
    "مرتفع": "#E4AFAA"
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def H(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def esc(x):
    if pd.isna(x):
        return ""
    return html.escape(str(x))


def normalize_month_name(month_name):
    text = str(month_name).strip()
    mapping = {
        "محرم": "محرم", "صفر": "صفر",
        "ربيع الأول": "ربيع الأول", "ربيع الاول": "ربيع الأول", "ربيع اول": "ربيع الأول",
        "ربيع الآخر": "ربيع الآخر", "ربيع الاخر": "ربيع الآخر", "ربيع الثاني": "ربيع الآخر",
        "جمادى الأولى": "جماد الأول", "جمادى الاولى": "جماد الأول",
        "جماد الأولى": "جماد الأول", "جماد الاولى": "جماد الأول",
        "جماد الأول": "جماد الأول", "جماد الاول": "جماد الأول",
        "جمادى الآخرة": "جماد الآخر", "جمادى الاخرة": "جماد الآخر",
        "جمادى الثانية": "جماد الآخر", "جمادى الثاني": "جماد الآخر",
        "جماد الآخرة": "جماد الآخر", "جماد الاخرة": "جماد الآخر",
        "جماد الآخر": "جماد الآخر", "جماد الاخر": "جماد الآخر",
        "جماد الثاني": "جماد الآخر",
        "رجب": "رجب", "شعبان": "شعبان", "رمضان": "رمضان", "شوال": "شوال",
        "ذو القعدة": "ذو القعدة", "ذو القعده": "ذو القعدة",
        "ذو الحجة": "ذو الحجة", "ذو الحجه": "ذو الحجة",
    }
    return mapping.get(text, text)


def format_number(value):
    try:
        if pd.isna(value):
            return "غير متاح"
        return f"{int(round(float(value))):,}"
    except Exception:
        return str(value)


def format_seasonal_count(value):
    try:
        if pd.isna(value):
            return "لا يوجد"
        value = float(value)
        if value <= 0:
            return "لا يوجد"
        return f"{int(round(value)):,}"
    except Exception:
        return "لا يوجد"


def normalize_level(level):
    text = str(level).strip()
    mapping = {
        "Low": "منخفض", "Medium": "متوسط", "High": "مرتفع", "Critical": "مرتفع",
        "low": "منخفض", "medium": "متوسط", "high": "مرتفع", "critical": "مرتفع",
        "منخفض": "منخفض", "متوسط": "متوسط", "مرتفع": "مرتفع",
        "حرج": "مرتفع", "عالي": "مرتفع", "مرتفع جدًا": "مرتفع",
    }
    return mapping.get(text, text)


def english_level(level):
    level = normalize_level(level)
    return {"منخفض": "Low", "متوسط": "Medium", "مرتفع": "High"}.get(level, "Medium")


def should_show_hajj_count(month_name, day):
    month_name = normalize_month_name(month_name)
    try:
        day = int(day)
    except Exception:
        return False
    if month_name == "ذو القعدة":
        return True
    if month_name == "ذو الحجة" and 1 <= day <= 9:
        return True
    return False


def should_show_tawaf_ifadah(month_name, day):
    month_name = normalize_month_name(month_name)
    try:
        day = int(day)
    except Exception:
        return False
    return month_name == "ذو الحجة" and day >= 10


def hex_to_rgba(hex_color, alpha=0.18):
    hex_color = str(hex_color).replace("#", "").strip()
    if len(hex_color) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def get_existing_model_path():
    for path in MODEL_PATHS:
        if path.exists():
            return path
    return None


def extract_hijri_day(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    nums = []
    current = ""
    for ch in text:
        if ch.isdigit():
            current += ch
        else:
            if current:
                nums.append(int(current))
                current = ""
    if current:
        nums.append(int(current))
    if not nums:
        return np.nan
    if nums[0] >= 1400 and len(nums) >= 3:
        return nums[2]
    if nums[-1] >= 1400 and len(nums) >= 3:
        return nums[0]
    return nums[0]


def classify_by_quantiles(series):
    series = pd.to_numeric(series, errors="coerce")
    q1 = series.quantile(0.33)
    q2 = series.quantile(0.66)

    def classify(x):
        if pd.isna(x):
            return "متوسط"
        if x <= q1:
            return "منخفض"
        elif x <= q2:
            return "متوسط"
        else:
            return "مرتفع"

    return series.apply(classify)


def normalize_columns(df):
    df = df.copy()
    rename_map = {
        "Hajj_Feature": "Hajj", "Tawaf_Ifadah_Feature": "Tawaf_Ifadah",
        "Tawaf_Ifadah_Featureh": "Tawaf_Ifadah", "Umrah_Count": "المعتمرين",
        "Visitors": "المعتمرين", "Predicted": "Prediction",
        "Prediction_Count": "Prediction", "predicted_visitors": "Prediction",
        "Gregorian Date": "Gregorian_Date", "Hijri Month": "Hijri_Month",
        "Hijri Day": "Hijri_Day"
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    if MONTH_COL in df.columns:
        df[MONTH_COL] = df[MONTH_COL].apply(normalize_month_name)
    return df


def add_display_cols(df):
    df = normalize_columns(df)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    else:
        df[DATE_COL] = pd.NaT
    if "Hijri_Date" not in df.columns:
        df["Hijri_Date"] = ""
    if "Hijri_Day" not in df.columns:
        df["Hijri_Day"] = df["Hijri_Date"].apply(extract_hijri_day)
    df["Hijri_Day_Num"] = pd.to_numeric(df["Hijri_Day"], errors="coerce")
    if df["Hijri_Day_Num"].isna().all():
        df["Hijri_Day_Num"] = df["Hijri_Date"].apply(extract_hijri_day)
    if "Weekday_AR" not in df.columns:
        weekday_map = {0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء", 3: "الخميس",
                       4: "الجمعة", 5: "السبت", 6: "الأحد"}
        if DATE_COL in df.columns:
            df["Weekday_AR"] = df[DATE_COL].dt.dayofweek.map(weekday_map)
        else:
            df["Weekday_AR"] = "غير محدد"
    if "AvgTemp_C" not in df.columns:
        df["AvgTemp_C"] = np.nan
    if "Hajj" not in df.columns:
        df["Hajj"] = 0
    if "Tawaf_Ifadah" not in df.columns:
        df["Tawaf_Ifadah"] = 0
    df["Hajj"] = pd.to_numeric(df["Hajj"], errors="coerce").fillna(0)
    df["Tawaf_Ifadah"] = pd.to_numeric(df["Tawaf_Ifadah"], errors="coerce").fillna(0)
    if "Prediction" not in df.columns:
        if TARGET_COL in df.columns:
            df["Prediction"] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        else:
            st.error("ملف المودل لا يحتوي على عمود Prediction أو عمود المعتمرين.")
            st.stop()
    df["Prediction"] = pd.to_numeric(df["Prediction"], errors="coerce")
    if "Crowding_Level" not in df.columns:
        df["Crowding_Level"] = classify_by_quantiles(df["Prediction"])
    else:
        df["Crowding_Level"] = df["Crowding_Level"].apply(normalize_level)
        valid = ["منخفض", "متوسط", "مرتفع"]
        invalid_mask = ~df["Crowding_Level"].isin(valid)
        if invalid_mask.any():
            fallback = classify_by_quantiles(df["Prediction"])
            df.loc[invalid_mask, "Crowding_Level"] = fallback.loc[invalid_mask]
    df = df.dropna(subset=[DATE_COL, "Hijri_Day_Num"])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


@st.cache_data
def load_data():
    model_path = get_existing_model_path()
    if model_path is None:
        st.error("لم يتم العثور على ملف المودل. تأكد أن الملف موجود باسم growth_rate_xgboost_model.pkl داخل مجلد models.")
        st.stop()
    try:
        package = joblib.load(model_path)
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل ملف المودل: {e}")
        st.stop()
    if isinstance(package, pd.DataFrame):
        return add_display_cols(package)
    if isinstance(package, dict):
        for key in ["test_df", "results_df", "df", "data", "predictions"]:
            if key in package and isinstance(package[key], pd.DataFrame):
                return add_display_cols(package[key])
    st.error("ملف المودل تم تحميله، لكن لا يحتوي على جدول نتائج مناسب.")
    st.stop()


def get_7_days(df, month, day):
    month = normalize_month_name(month)
    selected = df[
        (df[MONTH_COL].astype(str).str.strip() == str(month).strip()) &
        (df["Hijri_Day_Num"] == int(day))
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    selected = selected.sort_values(DATE_COL)
    start_date = selected.iloc[0][DATE_COL]
    if pd.isna(start_date):
        return pd.DataFrame()
    end_date = start_date + pd.Timedelta(days=6)
    out = df[(df[DATE_COL] >= start_date) & (df[DATE_COL] <= end_date)].copy()
    out = out.sort_values(DATE_COL).reset_index(drop=True)
    out["Local_Crowding_Level"] = classify_by_quantiles(out["Prediction"])
    out["Local_Crowding_Level"] = out["Local_Crowding_Level"].apply(normalize_level)
    return out


def get_available_days(df, month):
    month = normalize_month_name(month)
    days = (
        df[df[MONTH_COL].astype(str).str.strip() == str(month).strip()]["Hijri_Day_Num"]
        .dropna().astype(int).sort_values().unique().tolist()
    )
    return days


def get_best_day(df7, current_day, current_month):
    current_month = normalize_month_name(current_month)
    alt = df7[
        ~(
            (df7["Hijri_Day_Num"] == int(current_day)) &
            (df7[MONTH_COL].astype(str).str.strip() == str(current_month).strip())
        )
    ].copy()
    if alt.empty:
        return None
    return alt.sort_values("Prediction").iloc[0]


def get_decision(level):
    level = normalize_level(level)
    if level == "منخفض":
        return "مناسب للزيارة"
    elif level == "مرتفع":
        return "يفضل اختيار يوم آخر"
    else:
        return "مناسب مع الحذر"


def get_reason(decision):
    if decision == "مناسب للزيارة":
        return "من المتوقع أن يكون مستوى الازدحام منخفضًا مقارنة بالأيام القريبة، لذلك يعد هذا اليوم خيارًا مناسبًا لأداء العمرة."
    if decision == "يفضل اختيار يوم آخر":
        return "الازدحام المتوقع مرتفع مقارنةً بالأيام القريبة، لذلك يفضل اختيار يوم أقل ازدحامًا."
    return "الازدحام المتوقع ضمن المستوى المتوسط، لذلك يمكن أداء العمرة مع الحذر واختيار الوقت المناسب."


# ── Chart ───────────────────────────────────────────────────────────────────

def build_chart(df7, selected_month=None, selected_day=None):
    cdf = df7.copy().reset_index(drop=True)
    cdf["Prediction"] = pd.to_numeric(cdf["Prediction"], errors="coerce")
    cdf["Local_Crowding_Level"] = cdf["Local_Crowding_Level"].apply(normalize_level)
    cdf["x_label"] = (
        cdf["Weekday_AR"].astype(str) + "<br>" +
        cdf["Hijri_Day_Num"].astype(int).astype(str) + "-" +
        cdf[MONTH_COL].astype(str)
    )

    selected_mask = pd.Series(False, index=cdf.index)
    if selected_month is not None and selected_day is not None:
        selected_month = normalize_month_name(selected_month)
        selected_mask = (
            (cdf[MONTH_COL].astype(str).str.strip() == str(selected_month).strip()) &
            (cdf["Hijri_Day_Num"].astype(int) == int(selected_day))
        )

    y_min = cdf["Prediction"].min()
    y_max = cdf["Prediction"].max()
    pad = max((y_max - y_min) * 0.28, 500)
    base_y = max(0, y_min - pad)

    fig = go.Figure()

    # Fill areas
    for i in range(len(cdf) - 1):
        x0, x1 = cdf.loc[i, "x_label"], cdf.loc[i + 1, "x_label"]
        y0, y1 = cdf.loc[i, "Prediction"], cdf.loc[i + 1, "Prediction"]
        l1 = normalize_level(cdf.loc[i, "Local_Crowding_Level"])
        l2 = normalize_level(cdf.loc[i + 1, "Local_Crowding_Level"])
        seg_color = LEVEL_COLORS["مرتفع"] if "مرتفع" in [l1, l2] else (
            LEVEL_COLORS["متوسط"] if "متوسط" in [l1, l2] else LEVEL_COLORS["منخفض"]
        )
        fig.add_trace(go.Scatter(
            x=[x0, x0, x1, x1], y=[base_y, y0, y1, base_y],
            mode="lines", fill="toself",
            fillcolor=hex_to_rgba(seg_color, 0.14),
            line=dict(color="rgba(0,0,0,0)", width=0),
            hoverinfo="skip", showlegend=False
        ))

    # Line segments
    for i in range(len(cdf) - 1):
        x0, x1 = cdf.loc[i, "x_label"], cdf.loc[i + 1, "x_label"]
        y0, y1 = cdf.loc[i, "Prediction"], cdf.loc[i + 1, "Prediction"]
        l1 = normalize_level(cdf.loc[i, "Local_Crowding_Level"])
        l2 = normalize_level(cdf.loc[i + 1, "Local_Crowding_Level"])
        seg_color = LEVEL_COLORS["مرتفع"] if "مرتفع" in [l1, l2] else (
            LEVEL_COLORS["متوسط"] if "متوسط" in [l1, l2] else LEVEL_COLORS["منخفض"]
        )
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=seg_color, width=4, shape="spline"),
            hoverinfo="skip", showlegend=False
        ))

    # Points by level
    for level in ["منخفض", "متوسط", "مرتفع"]:
        level_df = cdf[cdf["Local_Crowding_Level"] == level].copy()
        fig.add_trace(go.Scatter(
            x=level_df["x_label"], y=level_df["Prediction"],
            mode="markers+text",
            marker=dict(size=14, color=LEVEL_COLORS[level], line=dict(color="white", width=2.5)),
            text=level_df["Prediction"].apply(
                lambda x: f"{int(round(float(x))):,}" if not pd.isna(x) else ""
            ),
            textposition="top center",
            textfont=dict(size=11, color="#053D33", family="Cairo"),
            hovertemplate="<b>%{x}</b><br>العدد المتوقع: %{y:,.0f}<extra></extra>",
            name=level
        ))

    # Selected day highlight
    if selected_mask.any():
        sp = cdf[selected_mask].iloc[[0]]
        fig.add_trace(go.Scatter(
            x=sp["x_label"], y=sp["Prediction"],
            mode="markers",
            marker=dict(size=44, color="rgba(196,144,63,0.14)",
                        line=dict(color="rgba(196,144,63,0.22)", width=2)),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=sp["x_label"], y=sp["Prediction"],
            mode="markers+text",
            marker=dict(size=22, color="#FFF8EC", line=dict(color="#C4903F", width=4)),
            text=["اليوم المختار"],
            textposition="bottom center",
            textfont=dict(size=11, color="#062B25", family="Cairo"),
            hovertemplate="<b>اليوم المختار</b><br>%{x}<br>العدد المتوقع: %{y:,.0f}<extra></extra>",
            showlegend=False
        ))

    q1 = cdf["Prediction"].quantile(0.33)
    q2 = cdf["Prediction"].quantile(0.66)
    fig.add_hline(y=q1, line_dash="dot", line_color="rgba(26,155,108,0.25)", line_width=1)
    fig.add_hline(y=q2, line_dash="dot", line_color="rgba(200,146,30,0.25)", line_width=1)

    fig.update_layout(
        height=380,
        margin=dict(l=16, r=16, t=32, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        font=dict(family="Cairo", size=12, color="#082E28"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0.0,
            title_text="مستوى الازدحام:",
            title_font=dict(size=11),
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0)",
            bordercolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(showgrid=False, title="",
                   tickfont=dict(family="Cairo", size=11)),
        yaxis=dict(
            range=[base_y, y_max + pad],
            showgrid=True, gridcolor="rgba(120,100,60,0.07)",
            tickformat=",",
            title=dict(text="عدد المعتمرين", font=dict(size=10))
        ),
        hoverlabel=dict(bgcolor="#052A24", bordercolor="#C4903F",
                        font=dict(color="#FFF8EC", family="Cairo", size=12))
    )
    return fig


# ── CSS ──────────────────────────────────────────────────────────────────────

H("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800;900&display=swap');

/* ─── Variables ─── */
:root {
  --g950: #021913;
  --g900: #052A24;
  --g800: #084038;
  --g700: #0D6B4F;
  --g600: #1A9B6C;
  --gold:  #C4903F;
  --gold2: #D8B870;
  --cream: #FAF6EE;
  --paper: rgba(255,255,255,0.88);
  --line:  rgba(196,144,63,0.28);
  --ink:   #0A2A24;
  --muted: #607068;
}

/* ─── Base ─── */
html, body, [class*="css"] {
  font-family: 'Cairo', sans-serif !important;
  direction: rtl;
}

#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }
.js-plotly-plot .plotly .modebar { display: none !important; }

.stApp {
  background:
    radial-gradient(ellipse at 8% 6%,  rgba(196,144,63,0.16) 0%, transparent 28%),
    radial-gradient(ellipse at 92% 10%, rgba(13,107,79,0.14)  0%, transparent 28%),
    radial-gradient(ellipse at 50% 98%, rgba(5,42,36,0.08)   0%, transparent 34%),
    linear-gradient(145deg, #FDFAF3 0%, #F5ECD8 50%, #FAF6EE 100%);
  color: var(--ink);
}

/* Subtle geometric texture */
.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: 0.055;
  background-image:
    linear-gradient(30deg,  rgba(5,42,36,0.5) 12%, transparent 12.5%, transparent 87%, rgba(5,42,36,0.5) 87.5%),
    linear-gradient(150deg, rgba(5,42,36,0.5) 12%, transparent 12.5%, transparent 87%, rgba(5,42,36,0.5) 87.5%),
    linear-gradient(30deg,  rgba(5,42,36,0.5) 12%, transparent 12.5%, transparent 87%, rgba(5,42,36,0.5) 87.5%),
    linear-gradient(150deg, rgba(5,42,36,0.5) 12%, transparent 12.5%, transparent 87%, rgba(5,42,36,0.5) 87.5%);
  background-size: 80px 140px;
  background-position: 0 0, 0 0, 40px 70px, 40px 70px;
  z-index: 0;
}
.stApp > * { position: relative; z-index: 1; }

.block-container {
  max-width: 1340px !important;
  padding-top: 0.75rem !important;
  padding-bottom: 2rem !important;
}

div[data-testid="stHorizontalBlock"] {
  gap: 1rem !important;
  align-items: stretch !important;
}

/* ─── Buttons ─── */
.stButton button {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #0D6B4F 0%, #052A24 100%) !important;
  color: #FFF8EC !important;
  border: 1px solid rgba(196,144,63,0.40) !important;
  border-radius: 14px !important;
  height: 48px !important;
  font-size: 14px !important;
  font-weight: 800 !important;
  letter-spacing: 0.1px;
  box-shadow: 0 12px 28px rgba(5,42,36,0.16), inset 0 1px 0 rgba(255,255,255,0.10) !important;
  transition: all .2s ease !important;
}
.stButton button::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(110deg, transparent 20%, rgba(255,255,255,0.12) 50%, transparent 80%);
  transform: translateX(110%);
  transition: .38s ease;
}
.stButton button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 20px 40px rgba(5,42,36,0.20) !important;
  filter: brightness(1.07);
}
.stButton button:hover::before { transform: translateX(-110%); }
.stButton button:focus, .stButton button:active {
  outline: none !important;
  box-shadow: 0 12px 28px rgba(5,42,36,0.16) !important;
}

/* ─── Header ─── */
.site-header {
  position: relative;
  overflow: hidden;
  min-height: 118px;
  padding: 22px 36px;
  margin-bottom: 16px;
  border-radius: 28px;
  border: 1px solid rgba(196,144,63,0.36);
  background:
    radial-gradient(circle at 15% 25%, rgba(216,184,112,0.20) 0%, transparent 28%),
    radial-gradient(circle at 85% 75%, rgba(26,155,108,0.18) 0%, transparent 28%),
    linear-gradient(135deg, #021913 0%, #084038 50%, #052A24 100%);
  box-shadow: 0 24px 58px rgba(5,42,36,0.18), inset 0 1px 0 rgba(255,255,255,0.08);
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
.site-header::before {
  content: "";
  position: absolute;
  inset: 0;
  opacity: 0.09;
  background-image:
    linear-gradient(60deg,  rgba(255,255,255,0.30) 1px, transparent 1px),
    linear-gradient(120deg, rgba(216,184,112,0.30) 1px, transparent 1px);
  background-size: 40px 40px;
}
.site-header::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(115deg, transparent 0%, rgba(255,255,255,0.07) 45%, transparent 78%);
  pointer-events: none;
}
.header-inner { position: relative; z-index: 2; }
.header-title {
  color: #FFF8EC;
  font-size: 32px;
  font-weight: 900;
  letter-spacing: -0.8px;
  line-height: 1.3;
  text-shadow: 0 8px 24px rgba(0,0,0,0.16);
}
.header-sub {
  color: #D8B870;
  font-size: 14px;
  font-weight: 700;
  margin-top: 8px;
  line-height: 1.7;
}
.header-rule {
  width: 260px;
  height: 1.5px;
  margin: 12px auto 0;
  background: linear-gradient(90deg, transparent, rgba(216,184,112,0.90), transparent);
}

/* ─── Nav ─── */
.nav-label {
  text-align: center;
  font-size: 10px;
  font-weight: 900;
  color: rgba(5,42,36,0.55);
  margin: -4px 0 8px 0;
  letter-spacing: 0.4px;
}

/* ─── Home page ─── */
.hero-box {
  position: relative;
  overflow: hidden;
  padding: 48px 44px 42px;
  margin-bottom: 20px;
  border-radius: 28px;
  text-align: center;
  background: linear-gradient(145deg, rgba(255,255,255,0.90), rgba(255,251,240,0.74));
  border: 1px solid rgba(196,144,63,0.24);
  box-shadow: 0 22px 54px rgba(5,42,36,0.07);
  backdrop-filter: blur(18px);
}
.hero-box::before {
  content: "";
  position: absolute;
  inset: 20px;
  border-radius: 20px;
  border: 1px solid rgba(196,144,63,0.11);
  pointer-events: none;
}
.hero-box::after {
  content: "";
  position: absolute;
  width: 280px;
  height: 280px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(13,107,79,0.09), transparent 68%);
  left: -100px;
  bottom: -130px;
  pointer-events: none;
}
.hero-badge {
  display: inline-block;
  padding: 7px 18px;
  border-radius: 999px;
  color: #0D6B4F;
  background: rgba(13,107,79,0.08);
  border: 1px solid rgba(13,107,79,0.15);
  font-size: 11.5px;
  font-weight: 900;
  margin-bottom: 16px;
}
.hero-title {
  color: #052A24;
  font-size: 40px;
  font-weight: 900;
  letter-spacing: -1.2px;
  line-height: 1.25;
}
.hero-gold {
  color: #9C6E18;
  font-size: 16px;
  font-weight: 800;
  margin-top: 10px;
}
.hero-body {
  color: #4A5A54;
  font-size: 14px;
  font-weight: 600;
  max-width: 760px;
  margin: 16px auto 0;
  line-height: 2;
}

.feat-grid { margin-bottom: 20px; }
.feat-card {
  position: relative;
  overflow: hidden;
  padding: 26px 22px 24px;
  border-radius: 22px;
  background: linear-gradient(145deg, rgba(255,255,255,0.88), rgba(255,249,232,0.70));
  border: 1px solid rgba(196,144,63,0.22);
  box-shadow: 0 18px 42px rgba(5,42,36,0.06);
  backdrop-filter: blur(14px);
  transition: transform .2s ease, box-shadow .2s ease;
  height: 100%;
}
.feat-card:hover { transform: translateY(-4px); box-shadow: 0 28px 60px rgba(5,42,36,0.10); }
.feat-card::before {
  content: "";
  position: absolute;
  top: 0; right: 0; left: 0;
  height: 3px;
  background: linear-gradient(90deg, #C4903F, rgba(13,107,79,0.50), transparent);
}
.feat-card::after {
  content: "";
  position: absolute;
  width: 80px;
  height: 80px;
  border-radius: 50%;
  left: -30px;
  bottom: -30px;
  background: rgba(196,144,63,0.08);
}
.feat-icon {
  width: 44px;
  height: 44px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  color: #9C6E18;
  background: rgba(216,184,112,0.14);
  border: 1px solid rgba(196,144,63,0.28);
  font-size: 14px;
  font-weight: 900;
  margin-bottom: 14px;
}
.feat-title { color: #052A24; font-size: 17px; font-weight: 900; }
.feat-desc  { color: #607068; font-size: 12.5px; font-weight: 600; line-height: 1.85; margin-top: 8px; }

/* ─── Form page ─── */
.form-card {
  position: relative;
  overflow: hidden;
  padding: 36px 40px;
  border-radius: 28px;
  background: linear-gradient(145deg, rgba(255,255,255,0.90), rgba(255,251,240,0.74));
  border: 1px solid rgba(196,144,63,0.26);
  box-shadow: 0 22px 54px rgba(5,42,36,0.07);
  backdrop-filter: blur(18px);
  margin-bottom: 20px;
}
.form-card::before {
  content: "";
  position: absolute;
  top: 0; right: 0; left: 0;
  height: 3px;
  background: linear-gradient(90deg, transparent, #C4903F, rgba(13,107,79,0.55), transparent);
}
.form-title {
  text-align: center;
  color: #052A24;
  font-size: 28px;
  font-weight: 900;
  letter-spacing: -0.6px;
}
.form-sub {
  text-align: center;
  color: #9C6E18;
  font-size: 13px;
  font-weight: 700;
  margin-top: 8px;
}
.form-rule {
  width: 260px;
  height: 1.5px;
  margin: 14px auto 20px;
  background: linear-gradient(90deg, transparent, #C4903F, transparent);
}
.form-section-badge {
  display: inline-block;
  padding: 8px 20px;
  border-radius: 999px;
  color: #0D6B4F;
  background: rgba(13,107,79,0.07);
  border: 1px solid rgba(13,107,79,0.13);
  font-size: 13px;
  font-weight: 900;
  margin-bottom: 18px;
}

.stTextInput label,
.stSelectbox label {
  color: #052A24 !important;
  font-weight: 800 !important;
  font-size: 13px !important;
}
.stTextInput input {
  min-height: 50px !important;
  border-radius: 14px !important;
  background: rgba(255,254,252,0.95) !important;
  border: 1.5px solid rgba(196,144,63,0.30) !important;
  box-shadow: 0 8px 20px rgba(5,42,36,0.04) !important;
  font-family: 'Cairo', sans-serif !important;
  font-size: 14px !important;
  color: #052A24 !important;
}
.stTextInput input:focus {
  border-color: rgba(13,107,79,0.55) !important;
  box-shadow: 0 0 0 3px rgba(13,107,79,0.10) !important;
}
.stSelectbox > div > div {
  min-height: 50px !important;
  border-radius: 14px !important;
  background: rgba(255,254,252,0.95) !important;
  border: 1.5px solid rgba(196,144,63,0.30) !important;
}

/* ─── Dashboard: Crowding hero card ─── */
.crowding-card {
  position: relative;
  overflow: hidden;
  min-height: 370px;
  padding: 38px 40px 34px;
  border-radius: 24px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  background:
    radial-gradient(circle at 18% 20%, rgba(255,255,255,0.10) 0%, transparent 26%),
    radial-gradient(circle at 80% 80%, rgba(0,0,0,0.12) 0%, transparent 30%),
    linear-gradient(145deg, #042F29 0%, #074840 46%, #031E1A 100%);
  border: 1px solid rgba(196,144,63,0.36);
  box-shadow: 0 20px 48px rgba(5,42,36,0.20), inset 0 1px 0 rgba(255,255,255,0.09);
  backdrop-filter: blur(10px);
}

/* Decorative geometric panel on right side */
.crowding-card::before {
  content: "";
  position: absolute;
  top: 0; right: 0;
  width: 35%;
  height: 100%;
  opacity: 0.10;
  background-image:
    linear-gradient(30deg,  rgba(196,144,63,0.60) 12%, transparent 12.5%, transparent 87%, rgba(196,144,63,0.60) 87.5%),
    linear-gradient(150deg, rgba(196,144,63,0.60) 12%, transparent 12.5%, transparent 87%, rgba(196,144,63,0.60) 87.5%),
    linear-gradient(30deg,  rgba(196,144,63,0.60) 12%, transparent 12.5%, transparent 87%, rgba(196,144,63,0.60) 87.5%),
    linear-gradient(150deg, rgba(196,144,63,0.60) 12%, transparent 12.5%, transparent 87%, rgba(196,144,63,0.60) 87.5%);
  background-size: 38px 66px;
  background-position: 0 0, 0 0, 19px 33px, 19px 33px;
  pointer-events: none;
}

/* Gold arc decoration */
.crowding-card::after {
  content: "";
  position: absolute;
  right: -70px;
  bottom: -110px;
  width: 400px;
  height: 240px;
  border-radius: 50%;
  border-top: 1.5px solid rgba(196,144,63,0.55);
  transform: rotate(-15deg);
  pointer-events: none;
}

.cc-inner { position: relative; z-index: 2; width: 100%; }

.cc-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 8px 20px;
  border-radius: 999px;
  color: #EAD49A;
  background: rgba(255,255,255,0.09);
  border: 1px solid rgba(255,255,255,0.16);
  font-size: 12px;
  font-weight: 900;
  margin-bottom: 18px;
  letter-spacing: 0.2px;
}
.cc-badge-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #C4903F;
  box-shadow: 0 0 8px 3px rgba(196,144,63,0.36);
  display: inline-block;
}

.cc-level {
  font-size: 82px;
  font-weight: 900;
  line-height: 0.95;
  letter-spacing: -1px;
  text-shadow: 0 14px 34px rgba(0,0,0,0.18);
  margin-bottom: 10px;
}

.cc-label {
  color: #D8B870;
  font-size: 16px;
  font-weight: 800;
  margin-bottom: 22px;
}

.cc-rule {
  width: 200px;
  height: 1px;
  margin: 0 auto 20px;
  background: linear-gradient(90deg, transparent, rgba(196,144,63,0.70), transparent);
}

.cc-stats {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
}

.cc-stat {
  padding: 10px 18px;
  border-radius: 14px;
  background: rgba(255,255,255,0.09);
  border: 1px solid rgba(255,255,255,0.13);
  min-width: 130px;
  text-align: center;
}
.cc-stat-label {
  color: rgba(255,248,236,0.62);
  font-size: 10.5px;
  font-weight: 700;
  margin-bottom: 4px;
}
.cc-stat-value {
  color: #FFF8EC;
  font-size: 18px;
  font-weight: 900;
  line-height: 1.2;
}

/* ─── Dashboard: Side metric cards ─── */
.metric-card {
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: row-reverse;
  align-items: center;
  gap: 16px;
  padding: 16px 20px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(255,249,232,0.80));
  border: 1px solid rgba(196,144,63,0.24);
  box-shadow: 0 12px 26px rgba(5,42,36,0.07), inset 0 -3px 0 rgba(196,144,63,0.30);
  min-height: 108px;
  height: 108px;
  backdrop-filter: blur(14px);
  transition: transform .2s ease;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card::before {
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  background: linear-gradient(180deg, var(--gold), rgba(13,107,79,0.40));
  border-radius: 2px;
}
.metric-icon {
  width: 58px;
  height: 58px;
  min-width: 58px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  font-weight: 900;
  color: #9C6E18;
  background: radial-gradient(circle, #FFFBF0, #FFF3D6);
  border: 1px solid rgba(196,144,63,0.22);
  box-shadow: 0 8px 20px rgba(196,144,63,0.12);
}
.metric-body { flex: 1; text-align: right; }
.metric-label { color: #607068; font-size: 12px; font-weight: 800; margin-bottom: 5px; }
.metric-value { color: #052A24; font-size: 26px; font-weight: 900; line-height: 1.18; letter-spacing: -0.4px; }
.metric-sub   { color: #607068; font-size: 11px; font-weight: 700; margin-top: 4px; }

/* ─── Recommendation box ─── */
.reco-card {
  position: relative;
  overflow: hidden;
  padding: 22px 28px 24px;
  border-radius: 20px;
  background: linear-gradient(145deg, rgba(255,255,255,0.94), rgba(255,250,238,0.82));
  border: 1px solid rgba(196,144,63,0.28);
  box-shadow: 0 14px 30px rgba(5,42,36,0.06);
  backdrop-filter: blur(14px);
  text-align: center;
}
.reco-card::before {
  content: "";
  position: absolute;
  top: 0; right: 0; left: 0;
  height: 3px;
  background: linear-gradient(90deg, transparent, #C4903F 30%, rgba(13,107,79,0.55) 70%, transparent);
} 
.reco-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 8px 18px;
  border-radius: 999px;
  color: #0D6B4F;
  background: rgba(13,107,79,0.08);
  border: 1px solid rgba(13,107,79,0.15);
  font-size: 12px;
  font-weight: 900;
  margin-bottom: 14px;
}

.reco-eyebrow { color: #7A6A4C; font-size: 11px; font-weight: 800; margin-bottom: 6px; letter-spacing: 0.3px; }
.reco-decision { font-size: 26px; font-weight: 900; margin-bottom: 8px; letter-spacing: -0.3px; }
.reco-rule {
  width: 160px;
  height: 1.5px;
  margin: 8px auto 12px;
  background: linear-gradient(90deg, transparent, #C4903F, transparent);
  position: relative;
}
.reco-rule::after {
  content: "◆";
  position: absolute;
  top: -8px;
  left: 50%;
  transform: translateX(-50%);
  color: #C4903F;
  font-size: 11px;
}
.reco-text { color: #2E3F3A; font-size: 13.5px; font-weight: 600; line-height: 1.9; max-width: 980px; margin: auto; }

/* ─── Suggest & chart section ─── */
.suggest-card {
  position: relative;
  overflow: hidden;
  padding: 26px 22px;
  border-radius: 22px;
  text-align: center;
  background:
    radial-gradient(circle at 18% 18%, rgba(196,144,63,0.16) 0%, transparent 30%),
    linear-gradient(160deg, #031E1A 0%, #074840 54%, #031E1A 100%);
  border: 1px solid rgba(196,144,63,0.32);
  box-shadow: 0 20px 48px rgba(5,42,36,0.18);
  backdrop-filter: blur(10px);
  min-height: 180px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.suggest-card::before {
  content: "";
  position: absolute;
  inset: 0;
  opacity: 0.08;
  background-image:
    linear-gradient(60deg,  rgba(255,255,255,0.30) 1px, transparent 1px),
    linear-gradient(120deg, rgba(196,144,63,0.30) 1px, transparent 1px);
  background-size: 32px 32px;
}
.suggest-inner { position: relative; z-index: 2; }
.suggest-title { color: #FFF8EC; font-size: 22px; font-weight: 900; letter-spacing: -0.4px; }
.suggest-sub   { color: #D8B870; font-size: 12px; font-weight: 700; line-height: 1.9; margin-top: 10px; }

.best-day-card {
  margin-top: 12px;
  padding: 18px 16px;
  border-radius: 18px;
  text-align: center;
  background: linear-gradient(145deg, rgba(238,249,243,0.98), rgba(255,255,255,0.92));
  border: 1px solid rgba(26,155,108,0.22);
  box-shadow: 0 14px 30px rgba(26,155,108,0.09);
  backdrop-filter: blur(12px);
}
.bdc-badge {
  display: inline-block;
  padding: 5px 13px;
  border-radius: 999px;
  color: #0D7A52;
  background: rgba(13,107,79,0.08);
  font-size: 10px;
  font-weight: 900;
  margin-bottom: 8px;
}
.bdc-day   { color: #052A24; font-size: 24px; font-weight: 900; line-height: 1.4; }
.bdc-meta  { color: #4E6158; font-size: 12px; font-weight: 700; line-height: 1.9; margin-top: 6px; }
.bdc-meta strong { color: #052A24; font-weight: 900; }
.bdc-note  { color: #0D6B4F; font-size: 11px; font-weight: 700; margin-top: 8px; line-height: 1.7; }

.chart-card {
  padding: 18px 20px 8px;
  border-radius: 22px;
  background: linear-gradient(145deg, rgba(255,255,255,0.88), rgba(255,250,238,0.70));
  border: 1px solid rgba(196,144,63,0.22);
  box-shadow: 0 14px 30px rgba(5,42,36,0.06);
  backdrop-filter: blur(14px);
}
.chart-title { color: #052A24; font-size: 15px; font-weight: 900; text-align: right; }
.chart-rule  { width: 80px; height: 1.5px; margin: 7px 0 4px auto; background: linear-gradient(90deg, #C4903F, transparent); }

/* ─── Info pill ─── */
.info-pill {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 10px 20px;
  border-radius: 999px;
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(196,144,63,0.28);
  color: #4E3F20;
  font-size: 12px;
  font-weight: 800;
  box-shadow: 0 10px 24px rgba(5,42,36,0.055);
  backdrop-filter: blur(12px);
}
.pill-tag {
  padding: 4px 11px;
  border-radius: 999px;
  background: rgba(13,107,79,0.09);
  color: #0D6B4F;
  font-size: 10.5px;
  font-weight: 900;
}

/* ─── Responsive ─── */
@media (max-width: 900px) {
  .header-title { font-size: 22px; }
  .cc-level { font-size: 52px; }
  .crowding-card { min-height: 260px; }
  .metric-card { height: auto; min-height: 90px; }
  .metric-value { font-size: 22px; }
  .suggest-card { min-height: 140px; }
}
</style>
""")


# ── Page components ──────────────────────────────────────────────────────────

def show_header():
    H("""
    <div class="site-header">
      <div class="header-inner">
        <div class="header-title">المنصة الذكية لتوقع الازدحام</div>
        <div class="header-sub">اختيار الوقت الأنسب لأداء العمرة</div>
        <div class="header-rule"></div>
      </div>
    </div>
    """)


def render_nav():
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("الرئيسية", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    with c2:
        if st.button("إدخال البيانات", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()
    with c3:
        if st.button("لوحة النتائج", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    H("<div style='height:14px;'></div>")


def crowding_hero_card(level, prediction_text, weekday, hijri_date):
    level = normalize_level(level)
    color = LEVEL_COLORS.get(level, "#1A9B6C")
    H(f"""
    <div class="crowding-card">
      <div class="cc-inner">
        <div class="cc-level" style="color:{color};">{esc(level)}</div>
        <div class="cc-label">مستوى الازدحام المتوقع</div>
        
      </div>
    </div>
    """)


def metric_card(label, value, sub, icon):
    H(f"""
    <div class="metric-card">
      <div class="metric-icon">{esc(icon)}</div>
      <div class="metric-body">
        <div class="metric-label">{esc(label)}</div>
        <div class="metric-value">{esc(value)}</div>
        <div class="metric-sub">{esc(sub)}</div>
      </div>
    </div>
    """)


def reco_card(decision, reason, level):
    level = normalize_level(level)
    color = LEVEL_COLORS.get(level, "#052A24")
    H(f"""
    <div class="reco-card">
        <div class="reco-badge">
            <span class="cc-badge-dot"></span>
        التوصية النهائية
    </div>
    <div class="reco-decision" style="color:{color};">{esc(decision)}</div>
    <div class="reco-rule"></div>
    <div class="reco-text">{esc(reason)}</div>
</div>
""")


def best_day_card(row):
    if row is None:
        H("""
        <div class="best-day-card">
          <div class="bdc-day">لا يوجد يوم بديل ضمن الأيام السبعة القريبة</div>
        </div>
        """)
        return
    day_name = esc(row.get("Weekday_AR", "غير متاح"))
    hijri_date = esc(row.get("Hijri_Date", ""))
    visitors = format_number(row.get("Prediction", 0))
    H(f"""
    <div class="best-day-card">
      <div class="bdc-badge">اليوم الأنسب للزيارة</div>
      <div class="bdc-day">{day_name}</div>
      <div class="bdc-meta">
        التاريخ الهجري: <strong>{hijri_date}</strong><br>
        العدد المتوقع: <strong>{visitors}</strong>
      </div>
      <div class="bdc-note">تم اختياره لأنه الأقل ازدحامًا ضمن الأيام السبعة القريبة.</div>
    </div>
    """)


def info_pill(text, tag):
    H(f"""
    <div class="info-pill">
      <span class="pill-tag">{esc(tag)}</span>
      <span>{esc(text)}</span>
    </div>
    """)


# ── Session state ────────────────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "home"
if "entered" not in st.session_state:
    st.session_state.entered = False
if "show_best_day" not in st.session_state:
    st.session_state.show_best_day = False


# ── Pages ────────────────────────────────────────────────────────────────────

def home_page():
    show_header()
    render_nav()

    H("""
    <div class="hero-box">
      <div class="hero-title">مرحباً بك في نظام التنبؤ بازدحام المعتمرين</div>
      <div class="hero-gold">Umrah Visitors Smart Forecasting System</div>
      <div class="hero-body">
        يعرض النظام العدد المتوقع للمعتمرين ومستوى الازدحام والتوصية المناسبة
        لمساعدتك في اختيار أنسب وقت لأداء العمرة.
      </div>
    </div>
    """)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("01", "مؤشر الازدحام", "عرض مستوى الازدحام لليوم المختار بشكل واضح وفوري."),
        ("02", "توقع عددي",    "تقدير العدد المتوقع للمعتمرين في اليوم المحدد."),
        ("03", "توصية ذكية",   "اقتراح يوم بديل أقل ازدحامًا عند الحاجة."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            H(f"""
            <div class="feat-card">
              <div class="feat-icon">{esc(icon)}</div>
              <div class="feat-title">{esc(title)}</div>
              <div class="feat-desc">{esc(desc)}</div>
            </div>
            """)

    H("<div style='height:22px;'></div>")
    _, mid, _ = st.columns([1, 1.05, 1])
    with mid:
        if st.button("بدء التنبؤ", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()


def input_page():
    show_header()
    render_nav()

    df_dates = load_data()

    H("""
    <div class="form-card">
      <div class="form-title">تحديد موعد العمرة</div>
      <div class="form-sub">اختر تاريخ الزيارة لمعرفة مستوى الازدحام المتوقع</div>
      <div class="form-rule"></div>

    </div>
    """)

    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("الاسم الكامل", placeholder="أدخل اسم المعتمر")
    with c2:
        nationality = st.selectbox("الجنسية", NATIONALITY_OPTIONS, index=0)

    c3, c4 = st.columns(2)
    with c3:
        month = st.selectbox("الشهر الهجري", HIJRI_MONTHS, index=10, key="month_selector")

    available_days = get_available_days(df_dates, month)
    if not available_days:
        available_days = list(range(1, 31))

    with c4:
        day = st.selectbox("التاريخ الهجري", available_days, index=0, key=f"day_selector_{month}")

    blocked = (month == "ذو القعدة" and nationality == "غير سعودي" and 1 <= int(day) <= 15)
    if blocked:
        st.error("لا يمكن عرض النتائج لهذا الاختيار؛ لأن الجنسية غير سعودي خلال الفترة من 1 إلى 15 ذو القعدة.")

    H("<div style='height:20px;'></div>")
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        submitted = st.button("عرض التوقع", use_container_width=True)

    if submitted:
        if not name.strip():
            st.warning("الرجاء إدخال الاسم الكامل.")
            return
        if blocked:
            return
        st.session_state.entered = True
        st.session_state.show_best_day = False
        st.session_state.selected_name = name.strip()
        st.session_state.selected_nationality = nationality
        st.session_state.selected_month = normalize_month_name(month)
        st.session_state.selected_day = int(day)
        st.session_state.page = "dashboard"
        st.rerun()


def dashboard_page():
    show_header()
    render_nav()

    if not st.session_state.entered:
        st.warning("الرجاء إدخال البيانات أولًا.")
        if st.button("الذهاب إلى إدخال البيانات"):
            st.session_state.page = "input"
            st.rerun()
        return

    month = normalize_month_name(st.session_state.selected_month)
    day   = int(st.session_state.selected_day)

    show_hajj   = should_show_hajj_count(month, day)
    show_tawaf  = should_show_tawaf_ifadah(month, day)

    df   = load_data()
    df7  = get_7_days(df, month, day)

    if df7.empty:
        st.error("لا توجد بيانات مطابقة لليوم أو الشهر المختار داخل ملف المودل.")
        return

    sel_df = df7[
        (df7[MONTH_COL].astype(str).str.strip() == str(month).strip()) &
        (df7["Hijri_Day_Num"] == int(day))
    ]
    if sel_df.empty:
        st.error("لا توجد بيانات لليوم المختار داخل ملف المودل.")
        return

    row        = sel_df.iloc[0]
    weekday    = row.get("Weekday_AR", "غير محدد")
    hijri_date = row.get("Hijri_Date", f"{day}-{month}")
    prediction = float(row.get("Prediction", 0))
    level      = normalize_level(row.get("Local_Crowding_Level", "متوسط"))
    decision   = get_decision(level)
    reason     = get_reason(decision)

    temp       = row.get("AvgTemp_C", np.nan)
    temp_text  = "غير متاحة" if pd.isna(temp) else f"{float(temp):.1f}°C"

    hajj_count  = format_seasonal_count(row.get("Hajj", 0))
    tawaf_count = format_seasonal_count(row.get("Tawaf_Ifadah", 0))
    best_day    = get_best_day(df7, day, month)
    pred_text   = format_number(prediction)

    # ── Top row: crowding card + 3 metric cards ──────────────────────────
    left_col, right_col = st.columns([1.6, 1], gap="large")

    with left_col:
        crowding_hero_card(level, pred_text, weekday, hijri_date)

    with right_col:
        metric_card("اليوم المختار",  weekday,   hijri_date, "📅")
        H("<div style='height:10px;'></div>")
        metric_card("العدد المتوقع",  pred_text, "معتمر",    "👥")
        H("<div style='height:10px;'></div>")
        metric_card("درجة الحرارة",   temp_text, "متوسط اليوم", "🌡")

    H("<div style='height:12px;'></div>")

    # ── Recommendation ───────────────────────────────────────────────────
    reco_card(decision, reason, level)

    # ── Seasonal info pills ──────────────────────────────────────────────
    if show_hajj or show_tawaf:
        H("<div style='height:10px;'></div>")
        _, s1, _ = st.columns([1, 1.4, 1])
        with s1:
            if show_hajj:
                info_pill(f"عدد الحجاج المتوقع: {hajj_count}", "حج")
            elif show_tawaf:
                info_pill(f"عدد طواف الإفاضة المتوقع: {tawaf_count}", "طواف")

    H("<div style='height:14px;'></div>")

    # ── Suggest + Chart ──────────────────────────────────────────────────
    sug_col, chart_col = st.columns([1, 2.3], gap="large")

    with sug_col:
        H("""
        <div class="suggest-card">
          <div class="suggest-inner">
            <div class="suggest-title">اقتراح يوم بديل</div>
            <div class="suggest-sub">
              يقارن النظام الأيام السبعة القريبة ويقترح
              الأقل ازدحامًا عند الحاجة.
            </div>
          </div>
        </div>
        """)

        H("<div style='height:10px;'></div>")
        if st.button("اقتراح يوم أنسب", use_container_width=True):
            st.session_state.show_best_day = True

        if st.session_state.show_best_day:
            best_day_card(best_day)

    with chart_col:
        H("""
        <div class="chart-card">
          <div class="chart-title">الاتجاه المتوقع خلال الأيام السبعة القريبة</div>
          <div class="chart-rule"></div>
        </div>
        """)
        fig = build_chart(df7, month, day)
        st.plotly_chart(fig, use_container_width=True)

    H("<div style='height:10px;'></div>")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        if st.button("إدخال تاريخ جديد", use_container_width=True):
            st.session_state.page = "input"
            st.session_state.show_best_day = False
            st.rerun()


# ── Router ───────────────────────────────────────────────────────────────────

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "input":
    input_page()
else:
    dashboard_page()
