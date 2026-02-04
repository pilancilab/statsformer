from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsformer.metalearning.stacking import OOFStacking, StackingType
from statsformer.data.dataset import Dataset
from statsformer.data.preselection import DatasetWithPreselection
from statsformer.llm.generated_prior import GeneratedPrior
from statsformer.llm.with_preselection import GeneratedPriorWithPreselection
from statsformer.metalearning.base import ModelAndKwargs
from statsformer.metalearning.metalearners import MetaLearnerType
from statsformer.models.base import CVMetrics, Model, ModelCV
from statsformer.prior import FeaturePrior, FeaturePriorSweepConfig
from statsformer.utils import plot_error_bars, plot_format_generator


BASELINE_COLORS = {
    "autogluon": "black",
    "automl_agent": "#FE6100",
    "lasso": "#888888",
    "no_llm": "#56B4E9",
    "rand_forest": "#117733",
    "llm_lasso": "#490092",
    "xgboost": "#A70C11",
    "lightgbm": "#009292"
}
OUR_COLORS = ["#DC267F"]


@dataclass
class MethodConfig(ABC):
    """
    Supervised learning method (either a baseline or statsformer configuration)
    to evaluate on a dataset.
    """
    name: str
    display_name: str

    @property
    @abstractmethod
    def is_baseline(self):
        pass


@dataclass
class Baseline(MethodConfig):
    """
    Configuration for a baseline supervised learning method.
    """
    model: Model

    @property
    def is_baseline(self):
        return True


@dataclass
class StatsformerConfig(MethodConfig):
    """
    Configuration for statsformer.
    """
    models: list[ModelAndKwargs] # Base models to include in the ensemble
    cv_k: int
    prior: GeneratedPrior | GeneratedPriorWithPreselection # LLM-generated scores
    feature_prior_sweep: FeaturePriorSweepConfig # How to transform LLM scores
    cv_metric: CVMetrics = field(default_factory=lambda: CVMetrics.MCC)
    preselection: DatasetWithPreselection | None = field(default=None) # Optional feature preselection
    metalearner_type: MetaLearnerType = field(default=MetaLearnerType.NONEG)
    stacking_type: StackingType = field(default=StackingType.BASE)
    oversample_cv: bool = field(default=False)

    @property
    def is_baseline(self):
        return False
    
    def __post_init__(self):
        self.sweep_beta = [
            model.inst().using_sample_weights() for model in self.models
        ]
    
    def get_oof_config_kwargs(self):
        return dict(
            k=self.cv_k,
            cv_metric=self.cv_metric,
            stacking_type=self.stacking_type,
            metalearner_type=self.metalearner_type,
            oversample_cv=self.oversample_cv
        )


@dataclass
class ExperimentRow:
    """
    Evalution of a single model on a single train/test split of a dataset.
    """
    name: str # For use in dataframe and output files
    display_name: str # For plotting
    is_baseline: bool
    train_ratio: float
    split_number: int
    accuracy: float | None = field(default=None)
    misclass: float | None = field(default=None)
    mse: float | None = field(default=None)
    mcc: float | None = field(default=None)
    auroc: float | None = field(default=None)


class BaseExperiment(ABC):
    """
    Abstract base class for experiments evaluating supervised learning methods
    on datasets. Automatically runs all models across all train/test splits, and
    saves results to disk. Also includes plotting utilities.
    """
    @property
    @abstractmethod
    def data(self) -> Dataset:
        pass

    @property
    @abstractmethod
    def results(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def base_output_dir(self, method_name=None) -> Path:
        pass

    @abstractmethod
    def set_results(self, df: pd.DataFrame):
        pass

    def _maybe_add_oof_models(
        self,
        oof_model: OOFStacking,
        models: list[ModelAndKwargs],
        feature_prior_sweep: FeaturePriorSweepConfig,
        prior: GeneratedPrior | GeneratedPriorWithPreselection,
        ratio_idx: int, split_idx: int
    ):
        if isinstance(prior, GeneratedPriorWithPreselection):
            scores = prior.get_scores_for_split(ratio_idx, split_idx)
        else:
            scores = prior.get_scores()

        if len(oof_model.model_configs) == 0 or \
                isinstance(prior, GeneratedPriorWithPreselection):
            oof_model.clear_models()
            for model in models:
                oof_model.add_model(
                    model=model,
                    priors=feature_prior_sweep,
                    scores=scores
                )

    @property
    def label_ours_in_plots(self):
        return True
    
    def get_seed(self, base_seed: int, ratio_idx: int, split_idx):
        return base_seed + ratio_idx * self.data.num_splits() + split_idx

    def _add_results(
        self, name: str,
        results: list[ExperimentRow],
    ):
        """
        Adds a set of ExperimentRow results to the overall results dataframe,
        saving to disk as well.
        """
        df = pd.DataFrame([asdict(x) for x in results])
        file = self.base_output_dir(name) / f"{name}.csv"
        df.to_csv(file, index=False)

        existing_df = self.results
        if existing_df is None:
            self.set_results(df)
        else:
            self.set_results(pd.concat((
                existing_df.drop(index=existing_df[existing_df["name"] == name].index),
                df,
            ), axis=0, copy=True, ignore_index=True))
    
    def _run_for_ratios_and_splits(
        self, fn, *args
    ) -> list[ExperimentRow]:
        """
        Runs a given function for all train/test splits and ratios
        of the dataset, returning a list of ExperimentRow results.
        """
        output = []
        dataset = self.data
        num_ratios, num_splits = dataset.num_ratios(), \
            dataset.num_splits()
        bar = tqdm(total=num_ratios * num_splits)

        tmpfile = Path(self.base_output_dir()) / "TMPFILE.csv"
        if tmpfile.exists():
            try:
                df = pd.read_csv(tmpfile)
                method = args[0]
                if (df["name"] == method.name).all():
                    output = [
                        ExperimentRow(**row) for row in df.to_dict(orient="records")
                    ]
            except BaseException as e:
                pass
        for ratio_idx in range(dataset.num_ratios()):
            for split_idx in range(dataset.num_splits()):
                if ratio_idx * dataset.num_splits() + split_idx < len(output):
                    bar.update()
                    continue
                output.append(fn(
                    *args, ratio_idx=ratio_idx,
                    split_idx=split_idx
                ))
                pd.DataFrame([asdict(x) for x in output]).to_csv(tmpfile, index=False)
                bar.update()
        return output
    
    def _run_configs_by_names(
        self, methods: list[MethodConfig],
        fn: callable,
        names: list[str],
        rerun=False
    ):
        """
        Runs a set of methods (either baselines or statsformer configurations)
        by name, skipping any that have already been run unless rerun=True.

        Calls _run_for_ratios_and_splits to run each method across all
        train/test splits and ratios of the dataset.
        """
        found = set()
        names = set(names)
        for method in methods:
            if method.name not in names:
                continue
            found.add(method.name)
            file = self.base_output_dir(method.name) / f"{method.name}.csv"
            if file.exists() and not rerun:
                print(f"[INFO] Method {method.display_name} already run. Skipping...")
                self.results.loc[self.results["name"] == method.name, "display_name"] = method.display_name
                continue
            print(f"[INFO] Running method {method.display_name}...")
            self._add_results(method.name, self._run_for_ratios_and_splits(
                fn, method
            ))
        not_found = names - found
        if not_found:
            print(f"[WARNING] Method(s) {list(names)} not found. Doing nothing.")
   
    def plot_win_ratio(
        self, our_method_name: str,
        their_method_name: str,
        metric_names: list[str],
        metric_display_names: list[str],
        save_filename: str=None
    ):
        df = self.results
        df = df[df["name"].isin([our_method_name, their_method_name])]
        our_display_name = df[df["name"] == our_method_name]["display_name"].iloc[0]
        their_display_name = df[df["name"] == their_method_name]["display_name"].iloc[0]

        ours_means = []
        theirs_means = []

        for metric_name in metric_names:
            pivot = df.pivot_table(
                index=["train_ratio", "split_number"],
                columns="name",
                values=metric_name
            ).dropna()

            if metric_name != "mse":
                ours_better = pivot[our_method_name] >= pivot[their_method_name]
                theirs_better = pivot[their_method_name] >= pivot[our_method_name]
            else: # mse is reversed
                ours_better = pivot[our_method_name] <= pivot[their_method_name]
                theirs_better = pivot[their_method_name] <= pivot[our_method_name]

            ours_means.append(ours_better.mean())
            theirs_means.append(theirs_better.mean())

        x = np.arange(len(metric_names))
        width = 0.35

        plt.figure(figsize=(3 * len(metric_names), 3))

        plt.bar(
            x - width / 2,
            ours_means,
            width,
            label=our_display_name,
            color="#4C72B0",
            edgecolor="black",
            linewidth=0.8
        )

        plt.bar(
            x + width / 2,
            theirs_means,
            width,
            label=their_display_name,
            color="#DD8452",
            edgecolor="black",
            linewidth=0.8
        )

        # Axes and labels
        plt.xticks(x, metric_display_names, fontsize=11)
        plt.ylabel("Ratio of Splits", fontsize=11)
        plt.title("Strict Win Ratio", fontsize=13)

        plt.ylim(0.0, 1.0)

        # Grid (y only)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Clean spines
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.legend(frameon=False, fontsize=11)
        plt.tight_layout()

        if save_filename is not None:
            plt.savefig(
                self.base_output_dir() / "plots" / save_filename,
                bbox_inches="tight",
                dpi=300
            )
            return str(self.base_output_dir() / "plots" / save_filename) + ".png"
        else:
            plt.show()

    def _plot_metric(
        self, metric: str,
        metric_display_name: str,
        extra_title: str="",
        error_bars: bool=False,
        bolded_method_names: list[str] | None=None,
        y_lim: tuple | None=None,
        custom_plot_name: str=None,
    ):
        """
        Plots a given metric across all methods evaluated in the experiment.

        This is the main plotting logic; specific metrics (accuracy, auroc, mse, etc.)
        call this function with appropriate parameters.
        """
        our_method_format = plot_format_generator(OUR_COLORS)

        all_method_names = self.results['name'].unique().tolist()
        baseline_color_names = BASELINE_COLORS.keys()
        unused_baseline_colors = [BASELINE_COLORS[color] for color in baseline_color_names
            if color not in all_method_names
        ]
        baseline_format = plot_format_generator(unused_baseline_colors)
        
        # Aggregate results across splits
        aggregated_results = (
            self.results
            .groupby(['name', 'train_ratio'], dropna=False)
            .agg(
                mean_metric=(metric, 'mean'),
                sd_metric=(metric, 'std'),
                qlow_metric=(metric, lambda x: x.quantile(0.1)),
                qhigh_metric=(metric, lambda x: x.quantile(0.9)),
                display_name=('display_name', lambda x: x.tolist()[0]),
                is_baseline=('is_baseline', lambda x: x.tolist()[0]),
                n=(metric, 'count'),
            ).reset_index()
        )
        aggregated_results["95ci"] = 1.96 * aggregated_results["sd_metric"] / (aggregated_results["n"] ** 0.5)

        # Separate our methods and baselines
        our_methods = aggregated_results[
            np.bitwise_not(aggregated_results['is_baseline'])
        ]['name'].unique().tolist()
        baseline_methods = aggregated_results[
            aggregated_results['is_baseline']
        ]['name'].unique().tolist()

        if bolded_method_names is None:
            bolded_method_names = our_methods
        bolded_method_names = set(bolded_method_names)
        
        # Main plotting logic
        fig, ax = plt.subplots(figsize=(16, 8))
        all_methods = baseline_methods + our_methods
        for (i, method) in enumerate(all_methods):
            data = aggregated_results[
                aggregated_results['name'] == method
            ]

            # Formatting based on whether baseline or our method
            is_baseline = data['is_baseline'].any()
            if is_baseline:
                if method in BASELINE_COLORS:
                    color = BASELINE_COLORS[method]
                    marker = 'o'
                else:
                    color, marker = next(baseline_format)
            else:
                color, marker = next(our_method_format)

            if method in bolded_method_names:
                linewidth = 3.5
                marker = '-D'
                markersize = 10
            else:
                linewidth = 2.5
                marker = f'-{marker}'
                markersize = 7

            label = data['display_name'].tolist()[0]
            if not is_baseline and self.label_ours_in_plots:
                label = f"{label} (Ours)"

            ax.plot(
                data['train_ratio'],
                data['mean_metric'],
                marker,
                linewidth=linewidth, color=color,
                markersize=markersize,
                label=label
            )

            if error_bars:
                plot_error_bars(
                    ax,
                    x=data['train_ratio'],
                    upper=data['mean_metric'] + data['95ci'],
                    lower=data['mean_metric'] - data['95ci'],
                    color=color,
                    x_offset=(i - len(all_methods)/2) *0.01
                )
        plt.grid(True, alpha=0.7)
        plt.ylabel(metric_display_name, fontdict={"size": 36})
        plt.xlabel("Train Ratio", fontdict={"size": 36})
        if y_lim:
            plt.ylim(*y_lim)
        plt.legend(fontsize=32, bbox_to_anchor=(1.02, 0.5), loc="center left")
        plt.tick_params(axis='both', labelsize=28)  # Change font size for both x and y axes
        title = f"{metric_display_name}: {self.data.display_name}"
        if extra_title:
            title = f"{title} ({extra_title})"
        plt.title(f"{title}\n", fontdict={"size": 40})
        
        (self.base_output_dir() / "plots").mkdir(exist_ok=True, parents=True)
        plotname = metric
        if error_bars:
            plotname += "_error_bars"
        if y_lim is not None:
            plotname += f"ylim_{y_lim[0]}_{y_lim[1]}"
        if custom_plot_name is not None:
            plotname = custom_plot_name
        fig.tight_layout()
        plt.savefig(
            self.base_output_dir() / "plots" / plotname,
            bbox_inches="tight",
            dpi=300
        )

        return str(self.base_output_dir() / "plots" / plotname) + ".png"

    def plot_accuracy(
        self,
        error_bars: bool=False,
        extra_title: str="",
        bolded_method_names: list[str] | None=None,
        y_lim: tuple | None=None,
        custom_plot_name: str=None,
    ):
        assert self.data.problem_type.is_classification()
        return self._plot_metric(
            metric="accuracy",
            metric_display_name="Accuracy",
            error_bars=error_bars,
            extra_title=extra_title,
            bolded_method_names=bolded_method_names,
            y_lim=y_lim,
            custom_plot_name=custom_plot_name
        )
    
    def plot_misclass(
        self,
        error_bars: bool=False,
        extra_title: str="",
        bolded_method_names: list[str] | None=None,
        y_lim: tuple | None=None,
        custom_plot_name: str=None,
    ):
        assert self.data.problem_type.is_classification()
        return self._plot_metric(
            metric="misclass",
            metric_display_name="Classification Error",
            error_bars=error_bars,
            extra_title=extra_title,
            bolded_method_names=bolded_method_names,
            y_lim=y_lim,
            custom_plot_name=custom_plot_name
        )
    
    def plot_auroc(
        self,
        error_bars: bool=False,
        extra_title: str="",
        bolded_method_names: list[str] | None=None,
        y_lim: tuple | None=None,
        custom_plot_name: str=None,
    ):
        assert self.data.problem_type.is_classification()
        return self._plot_metric(
            metric="auroc",
            metric_display_name="AUROC",
            error_bars=error_bars,
            extra_title=extra_title,
            bolded_method_names=bolded_method_names,
            y_lim=y_lim,
            custom_plot_name=custom_plot_name
        )
    
    def plot_mse(
        self,
        error_bars: bool=False,
        extra_title: str="",
        bolded_method_names: list[str] | None=None,
        y_lim: tuple | None=None,
        custom_plot_name: str=None,
    ):
        assert not self.data.problem_type.is_classification()
        return self._plot_metric(
            metric="mse",
            metric_display_name="Mean Sq. Error",
            error_bars=error_bars,
            extra_title=extra_title,
            bolded_method_names=bolded_method_names,
            y_lim=y_lim,
            custom_plot_name=custom_plot_name
        )


class Experiment(BaseExperiment):
    """
    Primary experiment class for evaluating baselines and statsformer
    configurations on a dataset.
    """
    def __init__(
        self,
        dataset: Dataset,
        base_output_dir: str | Path,
        baselines_output_dir: str | Path,
        baselines: list[Baseline],
        statsformer_configs: list[StatsformerConfig],
        num_threads: int=8,
        seed: int=42
    ):
        self.dataset = dataset
        self._base_output_dir = Path(base_output_dir)
        self._base_output_dir.mkdir(exist_ok=True, parents=True)

        self._baselines_output_dir = Path(baselines_output_dir)
        self._baselines_output_dir.mkdir(exist_ok=True, parents=True)

        self.baselines = baselines
        self.baseline_names = set([method.name for method in baselines])
        self.our_methods = statsformer_configs
        self.seed = seed
        self.num_threads = num_threads

        dfs = []
        for method in self.baselines + self.our_methods:
            file = self.base_output_dir(method.name) / f"{method.name}.csv"
            if file.exists():
                dfs.append(pd.read_csv(file))
        if dfs:
            self.df = pd.concat(dfs, axis=0, ignore_index=True)
        else:
            self.df = None
        
        self.oof_models = {config.name: OOFStacking(
            task=self.dataset.problem_type,
            num_threads=self.num_threads,
            **config.get_oof_config_kwargs()
        ) for config in statsformer_configs}

    @property
    def data(self):
        return self.dataset

    @property
    def results(self):
        return self.df
    
    def set_results(self, df):
        self.df = df
    
    def base_output_dir(self, method_name=None):
        if method_name is None:
            return self._base_output_dir
        return self._baselines_output_dir if method_name in self.baseline_names \
            else self._base_output_dir
    
    def _run_baseline_for_split(
        self,
        baseline: Baseline,
        ratio_idx: int,
        split_idx: int,
    ) -> ExperimentRow:
        data = self.dataset.get_split(ratio_idx, split_idx)
        baseline.model.set_num_threads(self.num_threads)
        seed = self.get_seed(self.seed, ratio_idx, split_idx)
        baseline.model.fit(
            data.X_train, data.y_train, random_seed=seed)
        return ExperimentRow(
            name=baseline.name,
            display_name=baseline.display_name,
            is_baseline=True,
            train_ratio=data.train_ratio,
            split_number=split_idx,
            **asdict(baseline.model.eval(data.X_test, data.y_test))
        )
    
    def _run_statsformer_for_split(
        self,
        config: StatsformerConfig,
        ratio_idx: int,
        split_idx: int,
    ) -> ExperimentRow:
        oof_model = self.oof_models[config.name]

        self._maybe_add_oof_models(
            oof_model=oof_model,
            models=config.models,
            feature_prior_sweep=config.feature_prior_sweep,
            prior=config.prior,
            ratio_idx=ratio_idx,
            split_idx=split_idx
        )

        if config.preselection is not None:
            data = config.preselection.get_split(ratio_idx, split_idx)
        else:
            data = self.dataset.get_split(ratio_idx, split_idx)

        seed = self.get_seed(self.seed, ratio_idx, split_idx)
        oof_model.fit(data.X_train, data.y_train, random_seed=seed)

        return ExperimentRow(
            name=config.name,
            display_name=config.display_name,
            is_baseline=False,
            train_ratio=data.train_ratio,
            split_number=split_idx,
            **asdict(oof_model.eval(data.X_test, data.y_test))
        )

    def run_baselines(self, rerun=False):
        self.run_baseline_by_names(self.get_baseline_names(), rerun)
    
    def run_baseline_by_names(self, names: list[str], rerun=False):
        self._run_configs_by_names(
            methods=self.baselines,
            fn=self._run_baseline_for_split,
            names=names, rerun=rerun
        )
    
    def get_baseline_names(self) -> list[str]:
        return [x.name for x in self.baselines]

    def run_statsformer(self, rerun=False):
        return self.run_statsformer_by_names(self.get_statsformer_names(), rerun)
    
    def run_statsformer_by_names(self, names: list[str], rerun=False):
        self._run_configs_by_names(
            methods=self.our_methods,
            fn=self._run_statsformer_for_split,
            names=names, rerun=rerun
        )
    
    def get_statsformer_names(self) -> list[str]:
        return [x.name for x in self.our_methods]
