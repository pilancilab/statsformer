from enum import Enum
from pathlib import Path
from statsformer.llm.generated_prior import GeneratedPrior
import numpy as np


class PermutedGeneratedPrior(GeneratedPrior):
    def __init__(
        self,
        generated_prior: GeneratedPrior,
        random_seed: int=42
    ):
        self.prior = generated_prior
        self.scores = generated_prior.get_scores()
        np.random.seed(random_seed)
        np.random.shuffle(self.scores)
    
    @classmethod
    def from_dir(
        cls,
        dir: str | Path,
        random_seed: int=42
    ): 
        return cls(
            GeneratedPrior.from_dir(dir),
            random_seed=random_seed
        )
    
    def get_scores(self) -> np.ndarray:
        return self.scores


class AdversarialTransformation(Enum):
    ONE_MINUS = "one_minus"
    INVERSE = "inverse"


class TransformedPrior(GeneratedPrior):
    def __init__(
        self,
        generated_prior: GeneratedPrior,
        transformation=AdversarialTransformation.ONE_MINUS,
    ):
        self.prior = generated_prior
        self.scores = generated_prior.get_scores()
        if transformation == AdversarialTransformation.ONE_MINUS:
            self.scores = 1 - self.scores
        elif transformation == AdversarialTransformation.INVERSE:
            self.scores = 1 / (self.scores + 1e-3)
        else:
            raise NotImplementedError(f"Unknown transformation {transformation}")
    
    @classmethod
    def from_dir(
        cls,
        dir: str | Path,
        random_seed: int=42
    ): 
        return cls(
            GeneratedPrior.from_dir(dir),
            random_seed=random_seed
        )
    
    def get_scores(self) -> np.ndarray:
        return self.scores