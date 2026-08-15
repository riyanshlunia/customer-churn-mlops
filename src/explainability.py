import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import shap
import joblib

from src.config import ARTIFACTS_DIR, DATASET_CONFIGS, RETENTION_PLAYBOOKS
from src.data_pipeline import DataPipeline


class ChurnExplainer:
    """Computes SHAP feature attributions and consulting retention recommendations."""

    def __init__(self, dataset_key: str = "telco", model_type: str = "lightgbm"):
        self.dataset_key = dataset_key
        self.model_type = model_type
        self.config = DATASET_CONFIGS[dataset_key]

        # Load trained model and feature names
        model_path = ARTIFACTS_DIR / f"{dataset_key}_{model_type}_model.joblib"
        names_path = ARTIFACTS_DIR / f"{dataset_key}_feature_names.joblib"

        if not model_path.exists() or not names_path.exists():
            raise RuntimeError("Model or feature names missing. Run model training first.")

        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(names_path)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_sample(self, X_proc_single: np.ndarray, raw_customer_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Compute SHAP values for a single processed sample vector."""
        if X_proc_single.ndim == 1:
            X_proc_single = X_proc_single.reshape(1, -1)

        shap_values = self.explainer.shap_values(X_proc_single)

        # Handle shape differences between SHAP outputs (binary vs multi-class / list vs array)
        if isinstance(shap_values, list):
            # Binary classification list: pick positive class (index 1)
            sv = np.array(shap_values[1])[0]
            base_val = float(self.explainer.expected_value[1])
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                sv = shap_values[0, :, 1]
                base_val = float(self.explainer.expected_value[1])
            else:
                sv = shap_values[0]
                base_val = float(self.explainer.expected_value)
        else:
            sv = np.squeeze(np.array(shap_values))
            base_val = float(self.explainer.expected_value)

        # Pair feature names with SHAP values
        attributions = []
        for name, val in zip(self.feature_names, sv):
            attributions.append({"feature": name, "shap_value": float(val)})

        # Sort by absolute SHAP impact
        attributions_sorted = sorted(attributions, key=lambda x: abs(x["shap_value"]), reverse=True)

        # Generate Actionable Retention Recommendations for top positive churn drivers
        recommendations = self.generate_recommendations(attributions_sorted, raw_customer_dict)

        return {
            "base_value": base_val,
            "attributions": attributions_sorted,
            "recommendations": recommendations,
        }

    def generate_recommendations(
        self, attributions: List[Dict[str, Any]], raw_customer_dict: Dict[str, Any]
    ) -> List[str]:
        """Generate tailored retention interventions based on top positive churn factors."""
        recs = []

        # Filter features that increase churn risk (positive SHAP value)
        risk_factors = [item for item in attributions if item["shap_value"] > 0]

        for item in risk_factors[:5]:
            feat = item["feature"]

            # Map encoded feature name to original raw feature
            for raw_feat, playbook in RETENTION_PLAYBOOKS.items():
                if raw_feat.lower() in feat.lower():
                    raw_val = str(raw_customer_dict.get(raw_feat, ""))

                    # Match specific condition if present
                    for key_cond, advice in playbook.items():
                        if key_cond.lower() == raw_val.lower() or key_cond.lower() in feat.lower():
                            if advice not in recs:
                                recs.append(f"**{raw_feat} ({raw_val})**: {advice}")
                        elif key_cond == "high" and (raw_val.replace('.', '', 1).isdigit() and float(raw_val) > 50):
                            if advice not in recs:
                                recs.append(f"**{raw_feat} ({raw_val})**: {advice}")
                        elif key_cond == "low" and (raw_val.replace('.', '', 1).isdigit() and float(raw_val) <= 50):
                            if advice not in recs:
                                recs.append(f"**{raw_feat} ({raw_val})**: {advice}")

        if not recs:
            recs.append("Customer shows low churn risk profile; continue standard loyalty engagement.")

        return recs


if __name__ == "__main__":
    pipeline = DataPipeline("telco")
    df = pipeline.load_raw_data()
    explainer = ChurnExplainer("telco", "lightgbm")

    sample_raw = df.iloc[0].to_dict()
    sample_proc = pipeline.transform(pd.DataFrame([sample_raw]))

    explanation = explainer.explain_sample(sample_proc[0], sample_raw)
    print("SHAP Base Value:", explanation["base_value"])
    print("\nTop 5 Attributions:")
    for item in explanation["attributions"][:5]:
        print(f"  {item['feature']}: {item['shap_value']:.4f}")

    print("\nRecommendations:")
    for r in explanation["recommendations"]:
        print(f"  - {r}")
