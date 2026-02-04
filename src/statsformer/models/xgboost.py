import numpy as np
from statsformer.models.base import ModelCV, ModelTask
import xgboost as xgb

from statsformer.prior import FeaturePrior
from statsformer.utils import clipped_logit


class XGBoost(ModelCV):
    def __init__(
        self,
        task: ModelTask,
        add_feature_weights: bool=True,
        add_instance_weights: bool=False,
        feature_weight_colsample_bynode: float=0.2,
        min_colsample_features: int=30,
        num_boost_round: int=50,
        num_threads: int=-1,
    ):
        self.model_task = task
        self.model_cv = None
        if task == ModelTask.BINARY_CLASSIFICATION:
            objective = 'binary:logistic'
        elif task == ModelTask.MULTICLASS:
            objective = 'multi:softmax'
        else:
            objective = 'reg:squarederror'
        self.model_params = dict(
            objective=objective,
            n_jobs=num_threads,
            nthread=num_threads,
            tree_method="hist",
            colsample_bynode=feature_weight_colsample_bynode
        )
        
        self.add_feature_weights = add_feature_weights
        self.add_instance_weights = add_instance_weights
        self.num_boost_round = num_boost_round
        self.num_threads = num_threads
        self.feature_weight_colsample_bynode = feature_weight_colsample_bynode
        self.min_colsample_features = min_colsample_features
    
    def task(self) -> ModelTask:
        return self.model_task
    
    def using_sample_weights(self):
        return self.add_instance_weights

    def set_num_threads(self, num_threads):
        self.num_threads = num_threads
        self.model_params['n_jobs'] = num_threads
        self.model_params["nthread"] = num_threads

    def _build_dmatrix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior
    ):
        kwargs = dict(label=y, nthread=self.num_threads)
        if self.add_feature_weights:
            kwargs["feature_weights"] = feature_prior.get_score()
        if self.add_instance_weights:
            kwargs["weight"] = feature_prior.get_sample_weights(X)
        
        return xgb.DMatrix(
            X, **kwargs
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior,
        random_seed: int=42,
    ) -> "XGBoost":
        dtrain = self._build_dmatrix(X, y, feature_prior)
        model_params = self.model_params
        model_params["seed"] = random_seed
        model_params["colsample_bynode"] = max(
            self.feature_weight_colsample_bynode,
            min(1, self.min_colsample_features / X.shape[1])
        )
            
        self.model = xgb.train(
            model_params, dtrain,
            num_boost_round=self.num_boost_round,
        )
        return self

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        if self.model_task.is_classification():
            return clipped_logit(self.model.predict(xgb.DMatrix(X, nthread=self.num_threads)))
        return self.model.predict(xgb.DMatrix(X, nthread=self.num_threads))


