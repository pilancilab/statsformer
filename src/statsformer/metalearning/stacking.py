from enum import Enum
from statsformer.metalearning.metalearners import MetaLearner
from sklearn.model_selection import BaseCrossValidator
from abc import ABC, abstractmethod
from statsformer.metalearning.base import ModelAndKwargs, ModelConfigCV, OneModelType
from statsformer.metalearning.metalearners import MetaLearnerType
from statsformer.models.base import CVMetrics, Model, ModelTask
from statsformer.prior import FeaturePrior, FeaturePriorSweepConfig
from statsformer.utils import get_cross_validator


import numpy as np


class StackingType(Enum):
    BASE = "base"
    GREEDY = "winner_takes_all"
    STRATIFIED = "stratified"
    PRESELECTED = "preselected"
    POSTSELECTED = "postselected"
    POSTSELECTED_STRATIFIED = "postselected_stratified"

    def inst(
        self,
        **kwargs
    ):
        if self == StackingType.BASE:
            return BaseStackingLearner(**kwargs)
        if self == StackingType.GREEDY:
            return WinnerTakesAll(**kwargs)
        if self == StackingType.STRATIFIED:
            return StratifiedStacker(**kwargs)
        if self == StackingType.PRESELECTED:
            return PreselectedStacker(**kwargs)
        if self == StackingType.POSTSELECTED:
            return PostselectedStacker(**kwargs)
        raise ValueError(f"Unknown stacking type: {self}")


class OOFStacking(Model):
    """
    The out-of-fold stacking ensemble model for statsformer. Fits multiple
    base models, and learns an ensemble over them using out-of-fold predictions.

    This is the main statsformer model.
    """
    def __init__(
        self,
        k: int,
        task: ModelTask,
        cv_metric: CVMetrics=CVMetrics.MCC, # MCC in theory is better for imbalanced data
        stacking_type: StackingType=StackingType.BASE, # base stacking works best
        preselected_stacker_num_models=2, # unused for base stacking
        oversample_cv: bool=False,
        metalearner_type=MetaLearnerType.NONEG, # non-negative works best
        metalearner_intercept: bool=False,
        num_threads: int=16,
    ):
        self.k = k
        self.model_task = task
        self.cv_metric = cv_metric
        self.intercept = 0
        self.oversample_cv = oversample_cv
        self.model_configs: list[OneModelType] = []
        self.best_model_pointer = None
        self.num_threads = num_threads

        self.stacking_type = stacking_type
        self.preselected_stacker_num_models = preselected_stacker_num_models
        self.meta_learner = metalearner_type.build(
            cv_metric=cv_metric,
            num_threads=num_threads,
            fit_intercept=metalearner_intercept,
            oversample_cv=oversample_cv
        )

    def task(self) -> ModelTask:
        return self.model_task

    def set_num_threads(self, num_threads):
        self.num_threads = num_threads
        self.meta_learner.set_num_threads(num_threads)
        for config in self.model_configs:
            config.base_model.model.set_num_threads(num_threads)
            for model in config.with_scores:
                model.model.set_num_threads(num_threads)

    def clear_models(self):
        self.model_configs = []

    def add_model(
        self,
        model: ModelAndKwargs,
        priors: list[FeaturePrior] | FeaturePriorSweepConfig,
        scores: np.ndarray=None
    ):
        """
        Add a model and associated priors to the ensemble. Each prior corresponds
        to a different model in the ensemble.
        """
        model.kwargs["task"] = self.task()
        model_inst = model.inst()
        if isinstance(priors, FeaturePriorSweepConfig):
            priors = priors.get_priors(
                scores,
                sweep_beta=model_inst.using_sample_weights()
            )
        new_config = OneModelType(
            base_model=ModelConfigCV(
                model=model_inst,
                priors = FeaturePrior.uniform(len(priors[0].feature_prior)),
            ),
            with_scores=[]
        )

        for p in priors:
            if p.temperature == 0:
                continue # base model already in bag
            model_inst = model.inst()
            new_config.with_scores.append(ModelConfigCV(
                model=model_inst,
                priors=p,
            ))
        self.model_configs.append(new_config)
        self.set_num_threads(self.num_threads)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior=None, # unused
        random_seed: int=42,
    ) -> "OOFStacking":
        """
        Fits the out-of-fold stacking ensemble by:
        1. Generating out-of-fold predictions for each base model,
        2. Learning weights over the base models using a meta-learner,
        3. Refitting each base model on the full data with the learned weights.
        4. If winner_takes_all is True, only the best model is kept.
        """
        cv = get_cross_validator(
            cv=self.k,
            y=y,
            is_classification=self.task().is_classification(),
            seed=random_seed
        )

        c = len(np.unique(y)) if self.task().is_classification() else 0
        self.out_dim = 1
        if self.task() == ModelTask.MULTICLASS:
            self.out_dim = c
        y = y.reshape(-1)

        stacker = self.stacking_type.inst(
            model_task=self.model_task,
            cv_metric=self.cv_metric,
            cv=cv,
            meta_learner=self.meta_learner,
            num_classes=c,
            cv_folds=self.k,
            random_seed=random_seed,
            num_models=self.preselected_stacker_num_models
        )

        self.intercept = stacker.fit(X, y, models=self.model_configs)

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Makes predictions using the fitted out-of-fold stacking ensemble.
        """
        y = np.zeros(
            (X.shape[0], self.out_dim) if self.out_dim > 1 else X.shape[0]
        )
        for config in self.model_configs:
            for model in ([config.base_model] + config.with_scores):
                if model.weight == 0:
                    continue
                y += model.weight * model.model.predict(X)
        if self.intercept:
            y += self.intercept
        return y


###############################################################################
# Stacking Learners
###############################################################################

class BaseStackingLearner(ABC):
    def __init__(
        self,
        model_task: ModelTask,
        cv_metric: CVMetrics,
        cv: BaseCrossValidator,
        meta_learner: "MetaLearner",
        cv_folds: int=5,
        num_classes: int=0,
        random_seed: int=42,
        **kwargs
    ):
        self.c = num_classes
        self.cv_metric = cv_metric
        self.cv = cv
        self.model_task = model_task
        self.random_seed = random_seed
        self.meta_learner = meta_learner
        self.k = cv_folds

    def _run_oof_for_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: ModelConfigCV,

    ):
        model.oof_predictions = model.model.fit_cv(
            X, y, model.priors, cv=self.cv,
            random_seed=self.random_seed,
        ).oof_predictions
        model.weight = 0

        model.cv_error = self.cv_metric.get_error_metric(y, model.oof_predictions, self.model_task)

    def _run_oof_for_learner(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: OneModelType,
    ):
        self._run_oof_for_model(X, y, model.base_model)
        for config in model.with_scores:
            self._run_oof_for_model(X, y, config)

    def _get_all_models(
        self, models: OneModelType | list[OneModelType]
    ) -> list[ModelConfigCV]:
        if isinstance(models, OneModelType):
            return [models.base_model] + models.with_scores
        return sum([self._get_all_models(model) for model in models], start=[])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: list[OneModelType]
    ) -> float: # returns the intercept if any
        for model in models:
            self._run_oof_for_learner(X, y, model)
        all_models = self._get_all_models(models)

        Z = np.stack([model.oof_predictions for model in all_models], axis=-1)
        meta_out = self.meta_learner.fit(
            Z, y, self.k, self.model_task,
            random_seed=self.random_seed + 12345
        )

        for (i, model_config) in enumerate(all_models):
            model_config.weight = meta_out.get_weight_for_method(i)

        return meta_out.intercept


class WinnerTakesAll(BaseStackingLearner):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: list[OneModelType]
    ):
        for model in models:
            self._run_oof_for_learner(X, y, model)
        all_models = self._get_all_models(models)
        best_model = min(all_models, key=lambda model: model.cv_error)
        best_model.weight = 1


class StratifiedStacker(BaseStackingLearner):
    def _fit_per_learner(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models_per_learner: list[list[ModelConfigCV]]
    ) -> tuple[list[np.ndarray], list[float]]:
        per_learner_predictions = []
        intercepts = []

        # Do OOF stacking for each learner
        for learner_models in models_per_learner:
            Z = np.stack([model.oof_predictions for model in learner_models], axis=-1)
            meta_out = self.meta_learner.fit(
                Z, y, self.k, self.model_task,
                random_seed=self.random_seed + 12345
            )
            for i, model in enumerate(learner_models):
                model.weight = meta_out.get_weight_for_method(i)

            pred = np.sum(Z * np.array([[
                meta_out.get_weight_for_method(i) for i in range(len(learner_models))
            ]]), axis=1) +  meta_out.intercept
            intercepts.append(meta_out.intercept)
            per_learner_predictions.append(pred)
        return per_learner_predictions, intercepts
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: list[OneModelType]
    ):
        for model in models:
            self._run_oof_for_learner(X, y, model)
        models_per_learner = [
            self._get_all_models(config) for config in models
        ]

        per_learner_predictions, intercepts = self._fit_per_learner(
            X, y, models_per_learner
        )
        
        # Do OOF stacking over the base learners
        Z = np.stack(per_learner_predictions, axis=-1)
        meta_out = self.meta_learner.fit(
            Z, y, self.k, self.model_task,
            random_seed=self.random_seed + 12345
        )

        intercept = meta_out.intercept if meta_out.intercept else 0
        for i in range(len(models)):
            for model in models_per_learner[i]:
                model.weight *= meta_out.get_weight_for_method(i)
            if intercepts[i]:
                intercept += intercepts[i]
        return intercept


class PreselectedStacker(BaseStackingLearner):
    def __init__(
        self,
        num_models: int=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_models = num_models
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: list[OneModelType]
    ):
        for model in self._get_all_models(models):
            model.weight = 0
        
        # Run all base models first and see which is best
        base_models = [model.base_model for model in models]
        for model in base_models:
            self._run_oof_for_model(X, y, model)

        # find the top two base models on the key model.cv_error
        best_model_idxs = sorted(
            range(len(base_models)), key=lambda m: base_models[m].cv_error
        )[:self.num_models]
        best_models = [models[i] for i in best_model_idxs]
        
        # now run oof stacking with just those models
        for config in best_models:
            for model in config.with_scores:
                self._run_oof_for_model(X, y, model)
        all_models = self._get_all_models(best_models)

        Z = np.stack([model.oof_predictions for model in all_models], axis=-1)
        meta_out = self.meta_learner.fit(
            Z, y, self.k, self.model_task,
            random_seed=self.random_seed + 12345
        )

        for (i, model_config) in enumerate(all_models):
            model_config.weight = meta_out.get_weight_for_method(i)
        return meta_out.intercept


class PostselectedStacker(StratifiedStacker):
    def __init__(
        self,
        num_models: int=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_models = num_models
    
    def _run_oof_and_get_best_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: list[OneModelType],
    ) -> list[OneModelType]:
        models_per_learner = [
            self._get_all_models(config) for config in models
        ]

        per_learner_predictions, _ = self._fit_per_learner(
            X, y, models_per_learner
        )

        per_learner_losses = [
            self.cv_metric.get_error_metric(y, preds, self.model_task)
            for preds in per_learner_predictions
        ]

        # find the top two base models on the key model.cv_error
        best_model_idxs = sorted(
            range(len(per_learner_losses)), key=lambda m: per_learner_losses[m]
        )[:self.num_models]
        return [models[i] for i in best_model_idxs]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: list[OneModelType]
    ):
        for model in models:
            self._run_oof_for_learner(X, y, model)
        
        best_models = self._run_oof_and_get_best_models(X, y, models)

        for model in self._get_all_models(models):
            model.weight = 0
        # now run oof stacking with just those models
        all_models = self._get_all_models(best_models)

        Z = np.stack([model.oof_predictions for model in all_models], axis=-1)
        meta_out = self.meta_learner.fit(
            Z, y, self.k, self.model_task,
            random_seed=self.random_seed + 12345
        )

        for (i, model_config) in enumerate(all_models):
            model_config.weight = meta_out.get_weight_for_method(i)
        return meta_out.intercept


class PostselectedStratifiedStacker(PostselectedStacker):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: list[OneModelType]
    ):
        for model in models:
            self._run_oof_for_learner(X, y, model)
        models_per_learner = [
            self._get_all_models(config) for config in models
        ]

        per_learner_predictions, intercepts = self._fit_per_learner(
            X, y, models_per_learner
        )

        per_learner_losses = [
            self.cv_metric.get_error_metric(y, preds, self.model_task)
            for preds in per_learner_predictions
        ]

        # find the top two base models on the key model.cv_error
        idxs = sorted(
            range(len(per_learner_losses)), key=lambda m: per_learner_losses[m]
        )
        
        best_model_idxs = idxs[:self.num_models]
        dropped_models = self._get_all_models(
            [models[i] for i in idxs[:self.num_models:]]
        )
        for model in dropped_models:
            model.weight = 0
        
        # Do OOF stacking over the base learners
        Z = np.stack([per_learner_predictions[i] for i in best_model_idxs], axis=-1)
        meta_out = self.meta_learner.fit(
            Z, y, self.k, self.model_task,
            random_seed=self.random_seed + 12345
        )

        intercept = meta_out.intercept if meta_out.intercept else 0
        for i in best_model_idxs:
            for model in models_per_learner[i]:
                model.weight *= meta_out.get_weight_for_method(i)
            if intercepts[i]:
                intercept += intercepts[i]
        return intercept