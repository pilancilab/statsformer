from statsformer.models.base import ModelTask
from statsformer.models.oversample_features import OversampleFeaturesModel

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RandomForest(OversampleFeaturesModel):
    def __init__(
        self,
        task: ModelTask,
        num_estimators: int=50,
        num_threads: int=8,
        use_sample_weights: bool=False,
        use_feature_weights: bool=True,
        oversample_proportion: float=1, # for feature weighting
    ):
        super().__init__(
            task=task,
            num_threads=num_threads,
            use_sample_weights=use_sample_weights,
            use_feature_weights=use_feature_weights,
            oversample_proportion=oversample_proportion
        )
        self.num_estimators = num_estimators
    
    def _fit_model(self, X, y, sample_weight, random_seed):
        if self.task().is_classification():
            self.model = RandomForestClassifier(
                n_estimators=self.num_estimators,
                n_jobs=self.num_threads,
                random_state=random_seed
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.num_estimators,
                n_jobs=self.num_threads,
                random_state=random_seed
            )
        self.model.fit(
            X, y.ravel(),
            sample_weight=sample_weight
        )
