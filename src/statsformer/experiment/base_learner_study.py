

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from statsformer.data.preselection import DatasetWithPreselection
from statsformer.llm.with_preselection import GeneratedPriorWithPreselection
from statsformer.metalearning.stacking import OOFStacking, StackingType
from statsformer.data.dataset import Dataset
from statsformer.experiment.base import BaseExperiment, ExperimentRow, MethodConfig
from statsformer.llm.generated_prior import GeneratedPrior
from statsformer.metalearning.base import ModelAndKwargs
from statsformer.metalearning.metalearners import MetaLearnerType
from statsformer.models.base import CVMetrics, Model, ModelCV
from statsformer.prior import FeaturePrior, FeaturePriorConfig, FeaturePriorSweepConfig, FunctionalForm


class BaseLearnerStudy(BaseExperiment):
    def __init__(
        self,
        dataset: Dataset | DatasetWithPreselection,
        model: ModelAndKwargs,
        prior: GeneratedPrior | GeneratedPriorWithPreselection,
        base_output_dir: str | Path,
        feature_prior_sweep: FeaturePriorSweepConfig,
        cv_k: int,
        cv_metric: CVMetrics=CVMetrics.MCC,
        num_threads: int=8,
        oversample_cv: bool=False,
        stacking_type: StackingType=StackingType.BASE,
        abbreviated: bool=False,
        seed=42,
    ):
        """
        Given a ModelCV and a GeneratedPrior, this class will run experiments
        for each feature prior configuration as well as an out-of-fold stacking
        model that uses all configurations as base learners
        """

        self.prior = prior
        self.cv_k = cv_k

        self.seed = seed
        self.num_threads = num_threads
        self.cv_metric = cv_metric

        self.model = model
        self.sweep_beta = model.inst().using_sample_weights()
        self.feature_prior_sweep = feature_prior_sweep
    
        oof_kwargs = dict(
            k=cv_k,
            task=dataset.problem_type,
            cv_metric=cv_metric,
            num_threads=num_threads,
            oversample_cv=oversample_cv
        )
        self.dataset = dataset

        if abbreviated:
            config = FeaturePriorConfig(
                functional_form=FunctionalForm.POWER,
                temperature=0
            )
            self.configs = [
                BaseLearnerStudyConfig(
                    model=model.inst(),
                    feature_prior_config=config,
                    name=_prior_config_to_string(config),
                    display_name=_prior_config_to_display_string(config),
                )
            ]
        else:
            self.configs = [
                BaseLearnerStudyConfig(
                    model=model.inst(),
                    feature_prior_config=config,
                    name=_prior_config_to_string(config),
                    display_name=_prior_config_to_display_string(config),
                ) for config in feature_prior_sweep.get_configs(self.sweep_beta)
            ]

        # Out-of-fold with a non-negativity constraint
        oof_model_pos = OOFStacking(
            stacking_type=stacking_type,
            metalearner_type=MetaLearnerType.NONEG,
            **oof_kwargs
        )

        self.configs.append(BaseLearnerStudyConfig(
            model=oof_model_pos,
            feature_prior_config=None,
            name="oof_positive",
            display_name="Out-of-fold Stacking",
            is_baseline_=False
        ))

        self._base_output_dir = Path(base_output_dir)
        self._base_output_dir.mkdir(exist_ok=True, parents=True)
        dfs = []
        
        for config in self.configs:
            file = self._base_output_dir / f"{config.name}.csv"
            if file.exists():
                dfs.append(pd.read_csv(file))
        if dfs:
            self.df = pd.concat(dfs, axis=0, ignore_index=True)
        else:
            self.df = None
    
    @property
    def label_ours_in_plots(self):
        return False

    @property
    def data(self):
        return self.dataset

    @property
    def results(self):
        return self.df
    
    def set_results(self, df):
        self.df = df
    
    def base_output_dir(self, method_name=None):
        return self._base_output_dir

    def _run_config_for_split(
        self,
        config: "BaseLearnerStudyConfig",
        ratio_idx: int,
        split_idx: int,
    ) -> ExperimentRow:
        if isinstance(config.model, OOFStacking):
            self._maybe_add_oof_models(
                oof_model=config.model,
                models=[self.model],
                feature_prior_sweep=self.feature_prior_sweep,
                prior=self.prior,
                ratio_idx=ratio_idx,
                split_idx=split_idx
            )
        if isinstance(self.prior, GeneratedPriorWithPreselection):
            scores = self.prior.get_scores_for_split(ratio_idx, split_idx)
        else:
            scores = self.prior.get_scores()
        if isinstance(self.dataset, DatasetWithPreselection) \
                    and not isinstance(config.model, OOFStacking) and \
                    config.feature_prior_config.temperature == 0:
            data = self.dataset.dataset.get_split(ratio_idx, split_idx)
            scores = np.ones(data.X_train.shape[1])
        else:
            data = self.dataset.get_split(ratio_idx, split_idx)
        config.model.set_num_threads(self.num_threads)
        seed = self.get_seed(self.seed, ratio_idx, split_idx)

        config.model.fit(
            data.X_train, data.y_train,
            feature_prior=FeaturePrior(
                feature_prior=scores,
                config=config.feature_prior_config
            ) if config.feature_prior_config else None,
            random_seed=seed
        )
        return ExperimentRow(
            name=config.name,
            display_name=config.display_name,
            is_baseline=config.is_baseline,
            train_ratio=data.train_ratio,
            split_number=split_idx,
            **asdict(config.model.eval(data.X_test, data.y_test))
        )
    
    def run_config_by_names(self, names: list[str], rerun=False):
        self._run_configs_by_names(
            methods=self.configs,
            fn=self._run_config_for_split,
            names=names, rerun=rerun
        )
    
    def run_all(self, rerun=False):
        self.run_config_by_names(self.get_config_names(), rerun=rerun)

    def get_config_names(self):
        return [config.name for config in self.configs]
    
    
@dataclass
class BaseLearnerStudyConfig(MethodConfig):
    model: ModelCV
    feature_prior_config: FeaturePriorConfig
    is_baseline_: bool=field(default=True)

    @property
    def is_baseline(self):
        return self.is_baseline_


def _prior_config_to_string(config: FeaturePriorConfig):
    default_config = FeaturePriorConfig(
        config.functional_form, config.temperature
    )
    val = f"{config.functional_form.value}_T_{config.temperature}"
    if config.sample_weight_beta != default_config.sample_weight_beta:
        val += f"_B_{config.sample_weight_beta}"
    if config.epsilon != default_config.epsilon:
        val += f"_E_{config.epsilon}"
    if config.power != default_config.power:
        val += f"_P_{config.power}"
    return val


def _prior_config_to_display_string(config: FeaturePriorConfig):
    default_config = FeaturePriorConfig(
        config.functional_form, config.temperature
    )
    if config.temperature == 0:
        return "Base Method"

    params = [f"α={config.temperature}"]
    if config.sample_weight_beta != default_config.sample_weight_beta:
        params.append(f"β={config.sample_weight_beta}")
    if config.epsilon != default_config.epsilon:
        params.append(f"ϵ={config.epsilon}")
    if config.power != default_config.power:
        params.append(f"P={config.power}")
    val = f"With Prior ({', '.join(params)})"
    return val
