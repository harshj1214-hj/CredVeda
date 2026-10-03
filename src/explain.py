import shap
import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ── constants ────────────────────────────────────────────────────────────────

_GOOD_TIERS = {"high", "good", "prime", "average", "1", "2"}
_BEST_TIERS = {"high", "good", "prime", "2"}

_DEFAULT_STEP_DENOMINATOR = 15_000.0   # splits income range into ~150 probes
_MIN_STEP = 100.0                       # floor so tiny rates don't infinite-loop


# ── shap contributions ────────────────────────────────────────────────────────

def extract_shap_contributions(
    model,
    background_data: pd.DataFrame,
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a DataFrame of SHAP feature impacts for the first row of *input_df*,
    sorted by absolute impact ascending (least influential first).

    Supports both binary classifiers (2-D shap values) and multi-class ones
    (3-D shap values), automatically slicing to the predicted class.
    """
    explainer = shap.TreeExplainer(model, data=background_data)
    shap_values = explainer(input_df)

    predicted_class = str(model.predict(input_df)[0])
    class_index = list(model.classes_).index(predicted_class)

    sv = shap_values.values
    # shape is (samples, features, classes) for multi-class; (samples, features) for binary
    impacts: NDArray = sv[0, :, class_index] if sv.ndim == 3 else sv[0]

    return (
        pd.DataFrame({
            "Feature":   list(input_df.columns),
            "Impact":    impacts,
            "AbsImpact": np.abs(impacts),
        })
        .sort_values("AbsImpact", ascending=True)
        .reset_index(drop=True)
    )


# ── counterfactual recourse ───────────────────────────────────────────────────

def _predict_tier(model, inputs: dict, feature_names: list[str]) -> str:
    """Run the model on *inputs* and return the predicted tier as a lowercase string."""
    row = pd.DataFrame([inputs])[feature_names]
    return str(model.predict(row)[0]).strip().lower()


def generate_counterfactual_recourse(
    current_inputs: dict,
    model,
    feature_names: list[str],
    currency_symbol: str = "₹",
    fx_rate: float = 1.0,
) -> str:
    """
    Explain what the applicant can do to reach a better risk tier.

    Strategy: incrementally reduce revolving liabilities (Disposable_Income up,
    DTI down) until the model flips to a better tier.  If no single-lever fix is
    found within the income ceiling, recommend a co-borrower instead.

    Args:
        current_inputs:  Feature dict for the applicant's current profile.
        model:           Trained sklearn-compatible classifier.
        feature_names:   Ordered feature list the model was trained on.
        currency_symbol: Symbol prepended to monetary amounts in the message.
        fx_rate:         Conversion factor from INR to the display currency.
                         1.0 keeps values in INR.

    Returns:
        A plain-English recourse string ready to show the applicant.
    """
    current_tier = _predict_tier(model, current_inputs, feature_names)

    if current_tier in _BEST_TIERS:
        return (
            "Your profile sits in the lowest risk tier. "
            "Maintain current credit utilization to stay here."
        )

    step = max(_MIN_STEP, _DEFAULT_STEP_DENOMINATOR / fx_rate)
    income_ceiling = float(current_inputs.get("Income", 500_000))

    probe = current_inputs.copy()

    for reduction in np.arange(step, income_ceiling, step):
        probe["Disposable_Income"] += reduction
        probe["DTI"] = max(0.0, probe["DTI"] - reduction / probe["Income"])

        new_tier = _predict_tier(model, probe, feature_names)

        if new_tier != current_tier and new_tier in _GOOD_TIERS:
            amount = f"{currency_symbol}{reduction:,.0f}"
            return (
                f"A prepayment of {amount} in revolving liabilities will move "
                f"your profile to Tier {new_tier.capitalize()}."
            )

    return (
        "Adding a co-borrower to increase total annual income is the "
        "most direct path to the next rating tier."
    )
