# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from src.utils.utils import load_object

app = Flask(__name__)

# Load artifacts once at startup
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"
MODEL_PATH = "artifacts/model.pkl"
preprocessor = load_object(PREPROCESSOR_PATH)
model = load_object(MODEL_PATH)

# Keep column names EXACTLY as the training pipeline expected
FEATURE_COLUMNS = ["carat","cut","color","clarity","depth","table","length","width","depth_mm"]

# Categorical options used in training 
CUT_OPTS     = ["Fair","Good","Very Good","Premium","Ideal"]
COLOR_OPTS   = ["D","E","F","G","H","I","J"]
CLARITY_OPTS = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]

def fmt_currency(x: float, symbol: str = "$") -> str:
    try:
        return f"{symbol}{x:,.2f}"
    except Exception:
        return f"{symbol}{x}"

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "form.html",
        cut_opts=CUT_OPTS,
        color_opts=COLOR_OPTS,
        clarity_opts=CLARITY_OPTS
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Pull values from form
        form = request.form
        payload = {
            "carat": float(form["carat"]),
            "cut": form["cut"],
            "color": form["color"],
            "clarity": form["clarity"],
            "depth": float(form["depth"]),
            "table": float(form["table"]),
            "length": float(form["length"]),
            "width": float(form["width"]),
            "depth_mm": float(form["depth_mm"]),
        }

        # Create DataFrame in the SAME column order used in training
        df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)

        # Transform + predict
        X = preprocessor.transform(df)            # shape: (1, 9)
        y_hat = float(model.predict(X)[0])        # raw currency (your target wasn't scaled)
        price_text = fmt_currency(y_hat, "$")     # change "$" to "₹" if you prefer

        # ---- Simple "key drivers" (top model importances) ----
        # Tree models: feature_importances_; Linear models: use |coef_|
        importances = None
        try:
            if hasattr(model, "feature_importances_"):
                vals = np.asarray(model.feature_importances_, dtype=float)
                importances = list(zip(FEATURE_COLUMNS, vals))
            elif hasattr(model, "coef_"):
                vals = np.abs(np.ravel(model.coef_)).astype(float)
                importances = list(zip(FEATURE_COLUMNS, vals))
        except Exception:
            importances = None

        top_drivers = None
        if importances is not None and len(importances) == len(FEATURE_COLUMNS):
            # sort high -> low and take top 3
            importances.sort(key=lambda t: t[1], reverse=True)
            top_drivers = [{"feature": f, "score": round(float(s), 4)} for f, s in importances[:3]]

        return render_template(
            "result.html",
            inputs=payload,
            prediction_text=price_text,
            drivers=top_drivers
        )
    except Exception as e:
        print("Prediction error:", e)
        return render_template("result.html", error="Could not generate prediction. Please check inputs.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
