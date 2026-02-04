import numpy as np
from statsformer.models.base import ModelCV


from dataclasses import dataclass, field
from typing import Type

from statsformer.prior import FeaturePrior


@dataclass
class ModelAndKwargs:
    model_class: Type[ModelCV]
    kwargs: dict = field(default_factory=dict)

    def inst(self):
        return self.model_class(**self.kwargs)


@dataclass
class ModelConfigCV:
    """
    Information for a model in statsformer's out-of-fold stacking ensemble.
    Includes the model itself, the feature prior used to fit it, and its
    weight in the ensemble.
    """
    model: ModelCV
    priors: FeaturePrior
    weight: float = field(default=1)
    cv_error: float = field(default=float('inf'))
    oof_predictions: np.ndarray | None = field(default=None)


@dataclass
class OneModelType:
    base_model: ModelConfigCV
    with_scores: list[ModelConfigCV]