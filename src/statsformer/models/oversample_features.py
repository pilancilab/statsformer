from abc import abstractmethod
import numpy as np

from statsformer.models.base import ModelCV, ModelTask
from statsformer.prior import FeaturePrior
from statsformer.utils import clipped_logit


def get_oversample_indices(feature_scores: np.ndarray, n_oversample: int):
    """
    feature_scores: non-negative array of shape (P,)
    n_oversample: number of *additional* feature indices to sample

    Returns:
        np.ndarray of indices (length n_oversample)
    """
    feature_scores = np.asarray(feature_scores, dtype=float).ravel()
    assert np.all(feature_scores >= 0)

    if feature_scores.sum() == 0:
        probs = np.ones_like(feature_scores) / len(feature_scores)
    else:
        probs = feature_scores / feature_scores.sum()

    return np.random.choice(
        len(feature_scores),
        size=n_oversample,
        replace=True,
        p=probs
    )


class OversampleFeaturesModel(ModelCV):
    def __init__(
        self,
        task: ModelTask,
        num_threads: int=8,
        use_sample_weights: bool=False,
        use_feature_weights: bool=True,
        oversample_proportion: float=1, # for feature weighting
    ):
        self.model_task = task
        self.num_threads = num_threads
        self.use_sample_weights = use_sample_weights
        self.use_feature_weights = use_feature_weights
        self.oversample_proportion = oversample_proportion
        self.feature_indices = []

    def task(self) -> ModelTask:
        return self.model_task

    def using_sample_weights(self):
        return self.use_sample_weights

    def set_num_threads(self, num_threads):
       self.num_threads = num_threads

    @abstractmethod
    def _fit_model(
        self, X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        random_seed: int
    ):
        pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior,
        random_seed: int=42,
    )-> "OversampleFeaturesModel":
        self.feature_indices = list(range(X.shape[1]))
        if self.use_feature_weights and feature_prior is not None and (feature_prior.temperature > 0):
            np.random.seed(random_seed)
            self.feature_indices += list(get_oversample_indices(
                feature_scores=feature_prior.get_score(),
                n_oversample=int(X.shape[1] * self.oversample_proportion)
            ))
        
        if self.use_sample_weights:
            sample_weight = feature_prior.get_sample_weights(X)
        else:
            sample_weight = np.ones(X.shape[0])

        # apply oversampling if feature weighting is enabled
        X = X[:, self.feature_indices]

        self._fit_model(X, y, sample_weight, random_seed)
        return self

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        X = X[:, self.feature_indices]
        if self.model_task.is_classification():
            return clipped_logit(self.model.predict_proba(X)[:, 1])
        return self.model.predict(X)