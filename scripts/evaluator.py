import os
import xgboost as xgb
import jax.numpy as jnp
from typing import Optional
import numpy as np

class BaseLineEvaluator:
    """
    Validates JAX-Forest predictions against the baseline XGBoost model.
    """
    def __init__(self, jax_predict_fn, pretrained_model_path: Optional[str] = "checkpoints/xgboost_model"):
        self.pretrained_model_path = pretrained_model_path
        self.jax_predict_fn = jax_predict_fn
        self.model = None
        self.model_base_score = None
        self.__init_model()

    def __init_model(self):
        self.model = xgb.XGBRegressor()
        self.model.load_model(os.path.join(self.pretrained_model_path, "xgboost_model.json"))
        # FIXME: as the model taken currently is Regressor so its ok to take first index for base score
        self.model_base_score = self.model.__dict__['base_score'][0]

    def check(self, X_sample, jax_params, single_tree: Optional[bool] = True, atol=1e-5):
        """
        Compares predictions and returns the error metrics.
        X_sample: JAX array or NumPy array of inputs.
        jax_params: The dictionary of JAX arrays (features, thresholds, etc.)
        atol: Absolute tolerance for floating point comparison.
        """
        xgb_preds = None
        # baseline XGBoost model expects NumPy/Pandas form
        X_np = np.array(X_sample)
        # ----- testing -------
        if single_tree:
            booster = self.model.get_booster()
            xgb_preds = booster.predict(xgb.DMatrix(X_np), iteration_range=(0, 1), output_margin=True)
        else:
            xgb_preds = self.model.predict(X_np)
        print(f"predictions from baseline xgboost mode --- {xgb_preds}")
        # now we try taking predictions from jaxified
        jax_preds = self.jax_predict_fn(X_sample, single_tree_inference=single_tree, **jax_params)
        # add in the computation for correct value to be matched with xgboost model
        jax_corrected_preds = self.model_base_score + jax_preds
        # if single_tree:
        #     jax_corrected_preds = self.model_base_score + jax_preds
        # else:
        #     jax_corrected_preds = jax_preds
        print(f"predictions from jaxified xgboost mode --- {jax_corrected_preds}")
        # lets calculate divergence with Maximum Absolute Error (MaxAE)
        diff = jnp.abs(xgb_preds - jax_corrected_preds)
        max_error = jnp.max(diff)
        mean_error = jnp.mean(diff)
        # check around tolerance
        is_consistent = max_error < atol

        status = "CONSISTENT" if is_consistent else "MISMATCH"
        print(f"[{status}] Max Error: {max_error:.2e}")
        
        if not is_consistent:
            print(f"XGB Sample: {xgb_preds[:2]}")
            print(f"JAX Sample: {jax_corrected_preds[:2]}")

        return {
            "is_consistent": bool(is_consistent),
            "max_error": float(max_error),
            "mean_error": float(mean_error),
            "xgb_sample_output": xgb_preds[:5],
            "jax_sample_output": jax_corrected_preds[:5]
        }
    
