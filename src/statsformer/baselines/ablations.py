from typing import Type
import numpy as np
from statsformer.llm.generated_prior import GeneratedPrior
from statsformer.metalearning.stacking import OOFStacking, StackingType
from statsformer.metalearning.base import ModelAndKwargs
from statsformer.models.base import CVMetrics, Model, ModelCV, ModelTask
from statsformer.models.glm import Lasso
from statsformer.prior import FeaturePriorSweepConfig, FunctionalForm


class LLMLassoBaseline(Model):
    def __init__(
        self,
        task: ModelTask,
        temperatures: list[float],
        prior: GeneratedPrior,
        cv_k: int=5,
        num_threads: int=8,
        cv_metric: CVMetrics=CVMetrics.ACC,
        **lasso_kwargs
    ):
        lasso_kwargs = {
            **lasso_kwargs,
            "default_folds_cv": cv_k,
            "num_threads": num_threads,
            "cv_metric": CVMetrics.LOSS,
            "task":  task
        }

        self.oof = OOFStacking(
            k=cv_k,
            task=task,
            num_threads=num_threads,
            stacking_type=StackingType.GREEDY,
            cv_metric=cv_metric
        )

        self.oof.add_model(
            ModelAndKwargs(
                model_class=Lasso,
                kwargs=lasso_kwargs
            ),
            FeaturePriorSweepConfig(
                functional_form=FunctionalForm.POWER,
                temperatures=temperatures,
                betas=[0]
            ).get_priors(prior.get_scores(), sweep_beta=False),
        )
        self.task_ = task

    def task(self):
        return self.task_

    def set_num_threads(self, num_threads):
        self.oof.set_num_threads(num_threads)

    def fit(self, X, y, feature_prior=None, random_seed = 42):
        self.oof.fit(X, y, random_seed=random_seed)
        return self

    def predict(self, X):
        return self.oof.predict(X)
    

class NoLLMAblation(Model):
    def __init__(
        self,
        task: ModelTask,
        nfeat: int,
        base_learners: list[ModelAndKwargs],
        cv_k: int=5,
        num_threads: int=8,
        cv_metric: CVMetrics=CVMetrics.MCC,
    ):
        self.oof = OOFStacking(
            k=cv_k,
            task=task,
            num_threads=num_threads,
            cv_metric=cv_metric
        )

        for model in base_learners:
            self.oof.add_model(
                model,
                FeaturePriorSweepConfig(
                    functional_form=FunctionalForm.POWER,
                    temperatures=[0],
                    betas=[0]
                ).get_priors(np.ones(nfeat))
            )
        self.task_ = task

    def task(self):
        return self.task_

    def set_num_threads(self, num_threads):
        self.oof.set_num_threads(num_threads)

    def fit(self, X, y, feature_prior=None, random_seed = 42):
        self.oof.fit(X, y, random_seed=random_seed)
        return self

    def predict(self, X):
        return self.oof.predict(X)