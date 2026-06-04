from pathlib import Path
import glob
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_FILE_NAME = "Intelligent System for Predicting Visitor Crowd Levels and Optimal Visiting Times at Al-Masjid Al-Haram.xlsx"

MODEL_PATH = Path("models/growth_rate_xgboost_model.pkl")
RESULTS_PATH = Path("outputs/growth_rate_xgboost_results_1446.xlsx")

DATE_COL = "Gregorian_Date"
YEAR_COL = "Hijri_Year"
MONTH_COL = "Hijri_Month"
TARGET_COL = "المعتمرين"


def find_data_file():
    exact_path = Path(DATA_FILE_NAME)

    if exact_path.exists():
        return exact_path

    matches = glob.glob(
        "Intelligent System for Predicting Visitor Crowd Levels and Optimal Visiting Times at Al-Masjid Al-Haram*.xlsx"
    )

    if matches:
        return Path(matches[0])

    raise FileNotFoundError(
        "Excel file not found. Please upload the Excel file with the expected name."
    )


def evaluate_model(name, actual, prediction):
    actual = pd.Series(actual).reset_index(drop=True)
    prediction = pd.Series(prediction).reset_index(drop=True)

    mae = mean_absolute_error(actual, prediction)
    rmse = np.sqrt(mean_squared_error(actual, prediction))
    mean_actual = actual.mean()

    mae_pct = (mae / mean_actual) * 100 if mean_actual else np.nan
    rmse_pct = (rmse / mean_actual) * 100 if mean_actual else np.nan

    mape = np.mean(
        np.abs((actual - prediction) / actual).replace([np.inf, -np.inf], np.nan)
    ) * 100

    return {
        "Model": name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Mean_Actual_Visitors": float(mean_actual),
        "MAE_Percentage": float(mae_pct),
        "RMSE_Percentage": float(rmse_pct),
        "MAPE": float(mape),
    }


def prepare_recommendation_event_features(df):
    """
    يحافظ على Hajj_Feature و Tawaf_Ifadah_Feature كقيم تفسيرية للتوصية فقط.
    لا يتم إضافتها إلى features الخاصة بتدريب المودل.
    """
    df = df.copy()

    if "Hajj_Feature" in df.columns:
        df["Hajj_Count_For_Recommendation"] = pd.to_numeric(
            df["Hajj_Feature"],
            errors="coerce"
        ).fillna(0)

        df["Hajj_Recommendation_Flag"] = np.where(
            df["Hajj_Count_For_Recommendation"] > 0,
            1,
            0
        )
    else:
        df["Hajj_Count_For_Recommendation"] = 0
        df["Hajj_Recommendation_Flag"] = 0

    if "Tawaf_Ifadah_Feature" in df.columns:
        df["Tawaf_Ifadah_Count_For_Recommendation"] = pd.to_numeric(
            df["Tawaf_Ifadah_Feature"],
            errors="coerce"
        ).fillna(0)

        df["Tawaf_Ifadah_Recommendation_Flag"] = np.where(
            df["Tawaf_Ifadah_Count_For_Recommendation"] > 0,
            1,
            0
        )
    else:
        df["Tawaf_Ifadah_Count_For_Recommendation"] = 0
        df["Tawaf_Ifadah_Recommendation_Flag"] = 0

    return df


def prepare_features(df):
    df = df.copy()

    required_cols = [DATE_COL, YEAR_COL, MONTH_COL, TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in Excel file: {missing_cols}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")

    df = df.dropna(subset=[DATE_COL, YEAR_COL, MONTH_COL, TARGET_COL])
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Keep Hajj and Tawaf for recommendation only
    df = prepare_recommendation_event_features(df)

    # Lag features
    df["lag_1"] = df[TARGET_COL].shift(1)
    df["lag_7"] = df[TARGET_COL].shift(7)
    df["lag_14"] = df[TARGET_COL].shift(14)
    df["lag_30"] = df[TARGET_COL].shift(30)

    # Rolling features
    df["rolling_mean_7"] = df[TARGET_COL].shift(1).rolling(7).mean()
    df["rolling_mean_14"] = df[TARGET_COL].shift(1).rolling(14).mean()
    df["rolling_mean_30"] = df[TARGET_COL].shift(1).rolling(30).mean()

    df["rolling_std_7"] = df[TARGET_COL].shift(1).rolling(7).std()
    df["rolling_std_14"] = df[TARGET_COL].shift(1).rolling(14).std()
    df["rolling_std_30"] = df[TARGET_COL].shift(1).rolling(30).std()

    # Difference features
    df["diff_1"] = df[TARGET_COL].shift(1) - df[TARGET_COL].shift(2)
    df["diff_7"] = df[TARGET_COL].shift(1) - df[TARGET_COL].shift(8)
    df["diff_14"] = df[TARGET_COL].shift(1) - df[TARGET_COL].shift(15)
    df["diff_30"] = df[TARGET_COL].shift(1) - df[TARGET_COL].shift(31)

    # Gregorian calendar features
    df["day"] = df[DATE_COL].dt.day
    df["month"] = df[DATE_COL].dt.month
    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    df["dayofyear"] = df[DATE_COL].dt.dayofyear

    # Ramadan feature
    df["is_ramadan"] = np.where(
        df[MONTH_COL].astype(str).str.contains(
            "رمضان|Ramadan|Ramadhan",
            case=False,
            na=False
        ),
        1,
        0
    )

    df["Season_Type"] = np.where(
        df["is_ramadan"] == 1,
        "Peak Season - Ramadan",
        "Normal Days"
    )

    optional_features = []

    # Weather temperature is used by model
    if "AvgTemp_C" in df.columns:
        df["AvgTemp_C"] = pd.to_numeric(df["AvgTemp_C"], errors="coerce")
        optional_features.append("AvgTemp_C")

    # Weather category is used by model
    if "Weather_Category" in df.columns:
        df["Weather_Category"] = df["Weather_Category"].astype(str).fillna("Unknown")

        weather_dummies = pd.get_dummies(
            df["Weather_Category"],
            prefix="Weather"
        ).astype(int)

        df = pd.concat([df, weather_dummies], axis=1)
        optional_features.extend(weather_dummies.columns.tolist())

    # Growth-rate target
    df = df[df["lag_7"] > 0].copy()

    df["growth_rate_7"] = (df[TARGET_COL] - df["lag_7"]) / df["lag_7"]
    df["growth_rate_7"] = df["growth_rate_7"].clip(lower=-1.0, upper=3.0)

    base_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
        "rolling_std_7",
        "rolling_std_14",
        "rolling_std_30",
        "diff_1",
        "diff_7",
        "diff_14",
        "diff_30",
        "day",
        "month",
        "dayofweek",
        "dayofyear",
        "is_ramadan",
    ]

    # Hajj and Tawaf are intentionally NOT included here
    features = base_features + optional_features

    df = df.dropna(
        subset=features + ["growth_rate_7", TARGET_COL]
    ).reset_index(drop=True)

    return df, features


def train_model(data_path=None):
    if data_path is None:
        data_path = find_data_file()
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Excel file not found: {data_path}")

    df_raw = pd.read_excel(data_path)
    df, features = prepare_features(df_raw)

    train_df = df[df[YEAR_COL] == 1445].copy()
    test_df = df[df[YEAR_COL] == 1446].copy()

    if train_df.empty:
        raise ValueError("Training data for Hijri year 1445 is empty.")

    if test_df.empty:
        raise ValueError("Testing data for Hijri year 1446 is empty.")

    X_train = train_df[features]
    y_train_growth = train_df["growth_rate_7"]

    X_test = test_df[features]
    y_test = test_df[TARGET_COL]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train_growth)

    test_df["Predicted_Growth_Rate_7"] = model.predict(X_test)
    test_df["Predicted_Growth_Rate_7"] = test_df["Predicted_Growth_Rate_7"].clip(
        lower=-1.0,
        upper=3.0
    )

    test_df["Prediction"] = test_df["lag_7"] * (
        1 + test_df["Predicted_Growth_Rate_7"]
    )

    test_df["Prediction"] = test_df["Prediction"].clip(lower=0)

    test_df["Absolute_Error"] = (
        test_df[TARGET_COL] - test_df["Prediction"]
    ).abs()

    test_df["Percentage_Error"] = (
        test_df["Absolute_Error"] / test_df[TARGET_COL]
    ) * 100

    test_df["Percentage_Error"] = test_df["Percentage_Error"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    results = evaluate_model(
        "Growth-Rate XGBoost",
        y_test,
        test_df["Prediction"]
    )

    results_df = pd.DataFrame([results]).round(4)

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    def season_metrics(group):
        mae = mean_absolute_error(group[TARGET_COL], group["Prediction"])
        rmse = np.sqrt(mean_squared_error(group[TARGET_COL], group["Prediction"]))
        mean_actual = group[TARGET_COL].mean()

        mape = np.mean(
            np.abs((group[TARGET_COL] - group["Prediction"]) / group[TARGET_COL]).replace(
                [np.inf, -np.inf],
                np.nan
            )
        ) * 100

        return pd.Series({
            "Number_of_Days": len(group),
            "Mean_Actual_Visitors": mean_actual,
            "MAE": mae,
            "RMSE": rmse,
            "MAE_Percentage": (mae / mean_actual) * 100 if mean_actual else np.nan,
            "RMSE_Percentage": (rmse / mean_actual) * 100 if mean_actual else np.nan,
            "MAPE": mape,
        })

    season_error_analysis = (
        test_df
        .groupby("Season_Type", group_keys=False)
        .apply(season_metrics)
        .reset_index()
        .round(4)
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(RESULTS_PATH, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Final_Test_Results", index=False)
        test_df.to_excel(writer, sheet_name="Final_Test_Predictions", index=False)
        season_error_analysis.to_excel(writer, sheet_name="Season_Error_Analysis", index=False)
        importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)

    package = {
        "model": model,
        "features": features,
        "results": results,
        "results_df": results_df,
        "season_error_analysis": season_error_analysis,
        "importance_df": importance_df,
        "test_df": test_df,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "data_path": str(data_path),
    }

    joblib.dump(package, MODEL_PATH)

    return package


if __name__ == "__main__":
    package = train_model()

    print("Model trained successfully.")
    print("Model saved to:", MODEL_PATH)
    print("Results saved to:", RESULTS_PATH)
    print("\nData file used:")
    print(package["data_path"])
    print("\nFeatures used by model:")
    print(package["features"])
    print("\nFinal Results:")
    print(package["results_df"].to_string(index=False))
