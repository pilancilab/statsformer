from dataclasses import asdict, dataclass
import json
import multiprocessing
from pathlib import Path

import pandas as pd
from statsformer.data.dataset import Dataset
from statsformer.data.preselection import DatasetWithPreselection
from statsformer.llm.generated_prior import MAX_BATCH_SIZE, GeneratedPrior
from statsformer.llm.common import LLMConfig
from statsformer.llm.pricing.base import LLMCost
from statsformer.llm.prompting import PriorGenerationPrompt
from statsformer.llm.rag.base import RAGConfig


class GeneratedPriorWithPreselection:
    def __init__(
        self,
        dataset: DatasetWithPreselection,
        base_prior: GeneratedPrior
    ):
        self.dataset = dataset
        self.base_prior = base_prior

    def get_scores_for_split(self, ratio_idx: int, split_idx: int):
        features = self.dataset.get_features(ratio_idx, split_idx)
        prior = pd.DataFrame(
            [self.base_prior.get_scores()],
            columns=self.dataset.dataset.feature_names()
        )
        return prior[features].to_numpy().ravel()


@dataclass
class GenerateForSplitArguments:
    prior_dir: str
    collect_reasoning: bool
    num_trials: int
    max_threads: int
    api_key: str | None
    
