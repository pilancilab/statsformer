

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from scipy.special import logit, expit

from statsformer.models.base import Model, ModelTask
from statsformer.utils import clipped_logit


class FeatureSelector(ABC):
    @abstractmethod
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_features: int,
        task: ModelTask,
        random_seed: int=42,
        **kwargs
    ) -> List[int]:
        pass


class RidgeWithFeatureSelectionBaseline(Model):
    """
    l2-penalized logistic regression for classification, ridge for regression.
    """
    def __init__(
        self,
        feature_selector: FeatureSelector,
        task: ModelTask,
        num_features: int=50,
        max_iter: int=100,
        Cs=10,
        num_folds: int=5,
        solver='lbfgs',
        scoring='balanced_accuracy',
        num_threads: int=8,

    ):
        self.num_threads = num_threads
        self.model_task = task
        self.feature_selector = feature_selector
        self.num_features = num_features
        self.selected_features = None

        if task.is_classification():
            self.kwargs = dict(
                max_iter=max_iter,
                Cs=Cs,
                cv=num_folds,
                solver=solver,
                scoring=scoring,
            )
        else:
            self.kwargs = dict(
                scoring=scoring,
                cv=num_folds,
            )

        self.scaler = StandardScaler()
    
    def task(self) -> ModelTask:
        return self.model_task
    
    def set_num_threads(self, num_threads):
        self.num_threads = num_threads
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior=None, # unused
        random_seed: int=42,
    ) -> "RidgeWithFeatureSelectionBaseline":
        selected_feature_indices = self.feature_selector.select_features(
            X=X,
            y=y,
            num_features=self.num_features,
            task=self.model_task,
            random_seed=random_seed
        )
        self.selected_features = selected_feature_indices
        X = X[:, selected_feature_indices]
        self.scaler.fit(X)
        X = self.scaler.transform(X, copy=True)

        if self.task().is_classification():
            self.model = LogisticRegressionCV(
                random_state=random_seed,
                n_jobs=self.num_threads,
                **self.kwargs
            )
        else:
            self.model = RidgeCV(
                random_state=random_seed,
                **self.kwargs
            )

        self.model.fit(X, y.reshape(-1))
        return self
    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        assert self.selected_features is not None, "Model must be fit before prediction"
        X = X[:, self.selected_features]
        X = self.scaler.transform(X, copy=True)

        if self.model_task.is_classification():
            return clipped_logit(self.model.predict_proba(X)[:, 1:2])
        return self.model.predict(X)

    def predict_class(self, X) -> np.ndarray:
        assert self.selected_features is not None, "Model must be fit before prediction"
        X = X[:, self.selected_features]
        X = self.scaler.transform(X, copy=True)

        if self.model_task.is_classification():
            return self.model.predict(X)
        raise NotImplementedError(
            f"predict_class not applicable for task {self.model_task}"
        )