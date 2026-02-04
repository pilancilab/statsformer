
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from lassonet import LassoNetClassifier, LassoNetRegressor

from statsformer.baselines.feature_selection import FeatureSelector
from statsformer.llm.generated_prior import GeneratedPrior
from statsformer.models.base import ModelTask


class XGBoostFeatureSelector(FeatureSelector):
    def __init__(self):
        pass

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_features: int,
        task: ModelTask,
        random_seed: int=42,
        **kwargs
    ) -> List[int]:
        if task.value == ModelTask.BINARY_CLASSIFICATION.value:
            objective = 'binary:logistic'
            model = xgb.XGBClassifier(
                objective=objective, random_state=random_seed).fit(X, y)   
        elif task.value == ModelTask.MULTICLASS.value:
            objective = 'multi:softmax'
            model = xgb.XGBClassifier(
                objective=objective, random_state=random_seed).fit(X, y)         
        else:
            objective = 'reg:squarederror'
            model = xgb.XGBRegressor(
                objective=objective, random_state=random_seed).fit(X, y)
        
        return list(np.argsort(-model.feature_importances_))[:num_features]


class MiFeatureSelector(FeatureSelector):
    def __init__(self):
        pass

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_features: int,
        task: ModelTask,
        **kwargs
    ) -> List[int]:
        mi = mutual_info_regression(X, y.flatten(), random_state=42, discrete_features=False)
        return list(np.argsort(-mi))[:num_features]


class RfeFeatureSelector(FeatureSelector):
    def __init__(
        self,
        max_iter: int=100,
        step: int=10
    ):
        self.max_iter = max_iter
        self.step = step

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_features: int,
        task: ModelTask,
        **kwargs
    ) -> List[int]:
        y = y.ravel()
        num_features = min(num_features, X.shape[1])
        if task.is_classification():
            model = LogisticRegression(max_iter=self.max_iter)
        else:
            model = LinearRegression()
        
        selector = RFE(model, n_features_to_select=num_features, step=self.step)
        selector = selector.fit(X, y)
        return list(np.where(selector.support_)[0])


class MrmrFeatureSelector(FeatureSelector):
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_features: int,
        task: ModelTask,
        random_seed: int=42,
        **kwargs
    ) -> List[int]:
        num_features = min(num_features, X.shape[1])
        y = y.ravel()
        d = X.shape[1]

        # Compute relevance once
        relevance = mutual_info_regression(
            X, y, random_state=random_seed, discrete_features=False,
        ) 

        selected = []
        remaining = np.arange(d).tolist()

        redundancy = np.zeros((num_features, d))

        for i in range(num_features):
            # compute MRMR score
            if i == 0:
                scores = relevance.copy()
            else:
                mean_redundancy = redundancy[:i].mean(axis=0)
                scores = relevance - mean_redundancy
                scores[selected] = -np.inf   # never pick again

            next_f = int(np.argmax(scores))
            selected.append(next_f)
            remaining.remove(next_f)

            # compute MI(X_j, X_next_f) for all remaining features
            if i < num_features - 1:
                mi_next = mutual_info_regression(
                    X[:, remaining], X[:, next_f],
                    discrete_features=False,
                    random_state=random_seed,
                )
                redundancy[i, remaining] = mi_next

        return selected


class RandomFeatureSelector(FeatureSelector):
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_features: int,
        task: ModelTask,
        random_seed: int=42,
        **kwargs
    ) -> List[int]:
        rng = np.random.default_rng(random_seed)
        return rng.choice(X.shape[1], size=num_features, replace=False).tolist()


class LassoNetFeatureSelector(FeatureSelector):
    def __init__(
        self,
        hidden_dims=(100,)
    ):
        self.hidden_dims = hidden_dims

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_features: int,
        task: ModelTask,
        random_seed: int=42,
        **kwargs
    ) -> List[int]:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y = y.ravel()

        # Initialize appropriate model
        if task.is_classification():
            model = LassoNetClassifier(
                hidden_dims=self.hidden_dims,
                verbose=0,
                random_state=random_seed,
                torch_seed=random_seed
            )
        else:
            model = LassoNetRegressor(
                hidden_dims=self.hidden_dims,
                verbose=0,
                random_state=random_seed,
                torch_seed=random_seed
            )
        model.fit(X_scaled.copy(), y.copy())

        best_model = None
        for m in model.path_:
            if m.selected.sum().item() > num_features:
                continue
            if best_model is None or best_model.val_loss > m.val_loss:
                best_model = m
        
        if best_model is None:# something went wrong
            return []
        return np.where(best_model.selected)[0]


class LLMScoreFeatureSelector(FeatureSelector):
    def __init__(
        self,
        prior: GeneratedPrior,
    ):
        self.prior = prior

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_features: int,
        task: ModelTask,
        **kwargs
    ) -> List[int]:
        scores = self.prior.get_scores()
        return list(np.argsort(-scores))[:num_features]