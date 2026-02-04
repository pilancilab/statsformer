import numpy as np
import pandas as pd
import random
import shutil
import tempfile
from lassonet import LassoNetClassifier, LassoNetRegressor
from sklearn.preprocessing import StandardScaler
from autogluon.tabular import TabularPredictor
from scipy.special import logit
import lightgbm as lgb

from statsformer.models.random_forest import RandomForest
from statsformer.models.glm import Lasso
from statsformer.models.base import Model, ModelTask

from statsformer.models.xgboost import XGBoost
from statsformer.prior import FeaturePrior
from statsformer.utils import clipped_logit


class XGBoostBaseline(XGBoost):
    def __init__(
        self,
        task: ModelTask,
        **xgboost_kwargs
    ):
        if "add_feature_weights" in xgboost_kwargs:
            del xgboost_kwargs["add_feature_weights"]
        if "add_instance_weights" in xgboost_kwargs:
            del xgboost_kwargs["add_instance_weights"]
        super().__init__(
            task=task,
            add_feature_weights=False,
            add_instance_weights=False,
            **xgboost_kwargs
        )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior=None, # unused
        random_seed: int=42,
    ) -> "XGBoostBaseline":
        return super().fit(
            X=X,
            y=y,
            feature_prior=FeaturePrior.uniform(X.shape[1]),
            random_seed=random_seed
        )


class LassoBaseline(Lasso):
    def __init__(
        self, **lasso_kwargs
    ):
        super().__init__(**lasso_kwargs)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior=None, # unused
        random_seed: int=42,
    ) -> "LassoBaseline":
        return super().fit(
            X=X,
            y=y,
            feature_prior=FeaturePrior.uniform(X.shape[1]),
            random_seed=random_seed
        )


class RandomForestBaseline(RandomForest):
    def __init__(
        self, **rf_kwargs
    ):
        super().__init__(**rf_kwargs)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior=None, # unused
        random_seed: int=42,
    ) -> "LassoBaseline":
        return super().fit(
            X=X,
            y=y,
            feature_prior=FeaturePrior.uniform(X.shape[1]),
            random_seed=random_seed
        )


class LassoNetBaseline(Model):
    def __init__(
        self,
        task: ModelTask,
        hidden_dims=(100,),
    ):
        self.model_task = task
        self.hidden_dims = hidden_dims

        self.scaler = None
    
    def task(self) -> ModelTask:
        return self.model_task
    
    def set_num_threads(self, num_threads):
        pass
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior=None, # unused
        random_seed: int=42,
    ) -> "LassoNetBaseline":
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X, copy=True)

        # Initialize appropriate model
        if self.task().is_classification():
            self.model = LassoNetClassifier(
                hidden_dims=self.hidden_dims,
                verbose=0,
                random_state=random_seed,
                torch_seed=random_seed
            )
        else:
            self.model = LassoNetRegressor(
                hidden_dims=self.hidden_dims,
                verbose=0,
                random_state=random_seed,
                torch_seed=random_seed
            )

        self.model.fit(X_scaled, y.ravel())
        return self
    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        X = self.scaler.transform(X, copy=True)

        if self.model_task is ModelTask.BINARY_CLASSIFICATION:
            return clipped_logit(self.model.predict_proba(X)[:, 1])
        elif self.model_task is ModelTask.MULTICLASS:
            raise NotImplementedError()
        return self.model.predict(X)


class AutoGluonBaseline(Model):
    def __init__(
        self,
        task: ModelTask,
        time_limit: int = 60,
        autogluon_path: str | None = None,
        **autogluon_kwargs
    ):
        self.model_task = task
        self.time_limit = time_limit
        self.autogluon_kwargs = autogluon_kwargs
        self.autogluon_path = autogluon_path
        self.model = None
        self.num_threads = -1
        self._temp_dir = None
    
    def task(self) -> ModelTask:
        return self.model_task
    
    def set_num_threads(self, num_threads):
        self.num_threads = num_threads
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior=None,
        random_seed: int=42,
    ) -> "AutoGluonBaseline":
        
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_df = pd.Series(y.ravel(), name="label")
        
        if self.task().is_classification():
            y_df = y_df.astype(str)
            problem_type = "binary" if self.task() == ModelTask.BINARY_CLASSIFICATION else "multiclass"
        else:
            problem_type = "regression"
        
        if self.autogluon_path is None:
            self._temp_dir = tempfile.mkdtemp(prefix="autogluon_")
            model_path = self._temp_dir
        else:
            model_path = self.autogluon_path
        
        predictor_kwargs = {
            "label": "label",
            "problem_type": problem_type,
            "verbosity": 0,
            "path": model_path,
            **self.autogluon_kwargs
        }
        
        self.model = TabularPredictor(**predictor_kwargs)
        train_data = pd.concat([X_df, y_df], axis=1)
        
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        self.model.fit(
            train_data,
            time_limit=self.time_limit,
            save_space=True,
        )
        
        return self
    
    def cleanup(self):
        if self._temp_dir is not None:
            try:
                shutil.rmtree(self._temp_dir)
            except (FileNotFoundError, OSError):
                pass
            finally:
                self._temp_dir = None
    
    def __del__(self):
        self.cleanup()
    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        if self.task().is_classification():
            if self.task() == ModelTask.BINARY_CLASSIFICATION:
                proba = self.model.predict_proba(X_df)
                if len(proba.columns) == 2:
                    if '1' in proba.columns:
                        pos_class = proba['1'].values
                    else:
                        pos_class = proba.iloc[:, -1].values
                return clipped_logit(pos_class)
            else:
                raise NotImplementedError("Multiclass classification not fully implemented yet")
        else:
            predictions = self.model.predict(X_df)
            return predictions.values


class LightGBMBaseline(Model):
    def __init__(
        self,
        task: ModelTask,
        num_estimators: int = 100,
        num_threads: int = 8,
        learning_rate: float = 0.1,
        max_depth: int = -1,
    ):
        self.model_task = task
        self.num_estimators = num_estimators
        self.num_threads = num_threads
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def task(self) -> ModelTask:
        return self.model_task

    def set_num_threads(self, num_threads: int):
        self.num_threads = num_threads

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior=None,
        random_seed: int = 42,
    ) -> "LightGBMBaseline":
        X = np.asarray(X)
        common_kwargs = dict(
            n_estimators=self.num_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_jobs=self.num_threads,
            random_state=random_seed,
            verbosity=-1,
        )

        if self.task().is_classification():
            self.model = lgb.LGBMClassifier(**common_kwargs)
        else:
            self.model = lgb.LGBMRegressor(**common_kwargs)

        self.model.fit(X, y.ravel())
        return self

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        np.asarray(X)
        if self.model_task.is_classification():
            # LightGBM returns probs directly
            return clipped_logit(self.model.predict_proba(X)[:, 1])
        return self.model.predict(X)
