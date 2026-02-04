from dataclasses import dataclass, field
from itertools import product
import numpy as np
from enum import Enum
from sklearn.preprocessing import StandardScaler


class FunctionalForm(Enum):
    POWER = "InverseImportance"


class FeatureMap(Enum):
    IDENTITY = "identity"
    SQRT = "sqrt"
    POWER = "power"
    EXP = "exp"


@dataclass
class FeaturePriorSweepConfig:
    """
    Configuration for transforming LLM-generated scores into feature and sample
    weights for downstream statistical models.
    """
    functional_form: FunctionalForm
    temperatures: list[float]
    betas: list[float] = field(default_factory=lambda: [0.5, 0.75, 1])
    sample_weight_power: int = field(default=1)
    sample_weight_epsilon: float = field(default=1e-4)
    feature_map: FeatureMap = field(default=FeatureMap.IDENTITY)
    sample_weight_map: FeatureMap = field(default=FeatureMap.IDENTITY)

    def get_priors(
        self, feature_prior: np.ndarray, sweep_beta: bool=False
    ):
        return [
            FeaturePrior(
                feature_prior=feature_prior,
                config=config
            ) for config in self.get_configs(sweep_beta)
        ]
    
    def get_configs(
        self, sweep_beta: bool=False
    ):
        betas = self.betas
        if not sweep_beta:
            betas = [0]
        
        non_zero_temp = [a for a in self.temperatures if a > 0]
        return [
            FeaturePriorConfig(
                functional_form=self.functional_form,
                temperature=0,
                feature_map=self.feature_map
            )
        ] + [
            FeaturePriorConfig(
                functional_form=self.functional_form,
                temperature=alpha,
                power=self.sample_weight_power,
                epsilon=self.sample_weight_epsilon,
                sample_weight_beta=beta,
                feature_map=self.feature_map
            ) for alpha, beta in product(non_zero_temp, betas)
        ]
        

@dataclass
class FeaturePriorConfig:
    """
    Configuration for transforming LLM-generated scores into feature and sample
    weights for downstream statistical models.
    """
    functional_form: FunctionalForm
    temperature: float
    power: int = field(default=1)
    epsilon: float = field(default=1e-4)
    exp_tau: float = field(default=1)
    sample_weight_beta: float = field(default=0)
    feature_map: FeatureMap = field(default=FeatureMap.IDENTITY)
    sample_weight_map: FeatureMap = field(default=FeatureMap.IDENTITY)


class FeaturePrior:
    def __init__(
        self,
        feature_prior: np.ndarray,
        config: FeaturePriorConfig,
    ):
        """
        Handles transforming LLM-generated feature importance scores into
        feature importance scores, penalties, and sample weights according to
        the specified functional form and temperature.
        """

        self.feature_prior = feature_prior
        if not FunctionalForm.__contains__(config.functional_form):
            raise NotImplementedError(
                f"Unknown functional form {config.functional_form}"
            )
        self.functional_form = config.functional_form
        assert 0 <= config.temperature, "Temperature must be non-negative"
        self.temperature = config.temperature
        self.config = config
    
    @classmethod
    def uniform(cls, num_features: int):
        """
        Returns a uniform feature prior (all ones), i.e., temperature = 0
        """
        return cls(
            feature_prior=np.ones(num_features),
            config=FeaturePriorConfig(
                functional_form=FunctionalForm.POWER,
                temperature=0,
            )
        )

    def get_score(
        self,
    ) -> np.ndarray:
        """
        Returns the feature importance scores according to the functional
        form and temperature.
        """
        alpha = self.temperature
        p = self.feature_prior
        if alpha == 0:
            s = np.ones_like(p)
        else:
            if self.config.feature_map == FeatureMap.IDENTITY:
                s = (p + self.config.epsilon) ** alpha
            elif self.config.feature_map == FeatureMap.SQRT:
                s = np.sqrt(p + self.config.epsilon) ** alpha
            elif self.config.feature_map == FeatureMap.POWER:
                s = (p + self.config.epsilon) ** (self.config.power * alpha)
            elif self.config.feature_map == FeatureMap.EXP:
                s = np.exp(self.config.exp_tau * alpha * p)
            else:
                raise ValueError(f"Unknown feature_map: {self.config.feature_map}")

        # normalize so average scale is ~1 (stabilizes gamma="scale")
        s = s / (np.mean(s) + self.config.epsilon)
        return s
    
    def get_penalty(
        self,
    ) -> np.ndarray:
        """
        Returns feature penalties (e.g., for Lasso) according to the functional
        form and temperature.
        """
        if self.functional_form == FunctionalForm.POWER:
            return (self.feature_prior + self.config.epsilon) ** (-self.temperature)
    
    def get_sample_weights(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Returns sample weights according to the feature scores and
        the configuration.
        """
        scaler = StandardScaler()
        X = np.abs(scaler.fit_transform(X.copy()))

        if self.temperature == 0:
            return np.ones(X.shape[0])
        feat_prior = self.get_score()[np.newaxis, :]
        X_hat = X ** self.config.power
        sample_score = np.sum(
            feat_prior * X_hat, axis=1
        ) / (np.linalg.norm(feat_prior) * np.linalg.norm(X_hat, axis=1))
        min_s = np.min(sample_score)
        max_s = np.max(sample_score)
        w_tilde = (sample_score - min_s) / (max_s - min_s + self.config.epsilon)

        if self.config.sample_weight_map == FeatureMap.IDENTITY:
            return (1 - self.config.sample_weight_beta) + \
                self.config.sample_weight_beta * w_tilde
        if self.config.sample_weight_map == FeatureMap.POWER:
            return (w_tilde) ** self.config.sample_weight_beta
        if self.config.sample_weight_map == FeatureMap.EXP:
            return np.exp(w_tilde * self.config.sample_weight_beta)
        raise NotImplementedError(f"Feature map {self.config.feature_map} not implemented for sample weights")