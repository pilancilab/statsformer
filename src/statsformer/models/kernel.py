import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from statsformer.models.base import ModelCV, ModelTask
from statsformer.prior import FeaturePrior
from statsformer.utils import clipped_logit

# Core idea: scale each feature j by s_j = g(p_j) where p_j is the
# feature importance from the prior, and g is some mapping (e.g., sqrt).
# Then train a standard kernel SVM on the scaled features (X tilde = X * s).

class WeightedKernelSVM(ModelCV):
    def __init__(
        self,
        task: ModelTask,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        seed: int = 42,
    ):
        self.model_task = task
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self._feature_scale = None

        if task.is_classification():
            self.model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
                probability=True,
                random_state=seed,
            )
        else:
            self.model = SVR(
                kernel=kernel,
                C=C,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
            )

    def task(self) -> ModelTask:
        return self.model_task

    def using_feature_weights(self):
        return True

    def set_num_threads(self, num_threads):
        # libsvm doesn't support n_jobs; keep interface compatibility
        return

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._feature_scale is None:
            raise RuntimeError("WeightedKernelSVM must be fit before calling predict().")
        Xs = self.scaler.transform(X)
        return Xs * self._feature_scale


    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior,
        random_seed: int = 42,
    ) -> "WeightedKernelSVM":
        # Fit scaler on training fold only
        self.scaler.fit(X)

        # Pull feature prior scores exactly like RF pulls sample weights
        p = feature_prior.get_score()
        p = np.asarray(p, dtype=float).reshape(-1)
        assert p.shape[0] == X.shape[1], f"feature prior dim {p.shape[0]} != X dim {X.shape[1]}"

        self._feature_scale = p
        Xt = self._transform(X)

        # Ensure deterministic behavior where possible
        if hasattr(self.model, "random_state"):
            self.model.random_state = random_seed

        self.model.fit(Xt, y.ravel())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xt = self._transform(X)
        if self.model_task.is_classification():
            return clipped_logit(self.model.predict_proba(Xt)[:, 1])
        return self.model.predict(Xt)
