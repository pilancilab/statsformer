from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

from sklearn.metrics import roc_auc_score
from scipy.special import expit
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import BaseCrossValidator
from statsformer.prior import FeaturePrior
from statsformer.utils import get_cross_validator, get_oversample_indices


class ModelTask(Enum):
    """
    Encodes whether a task is binary classification, multiclass classification,
    or regression
    """
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

    def is_classification(self):
        return self.value == ModelTask.BINARY_CLASSIFICATION.value \
            or self.value == ModelTask.MULTICLASS.value


@dataclass
class EvalResult:
    """
    All evaluation metrics that might be computed for a model. Depending on
    the task, only a subset will be populated (e.g., AUROC does not exist for
    regression).
    """
    accuracy: float | None = field(default=None)
    misclass: float | None=field(default=None)
    auroc: float | None = field(default=None)
    mcc: float | None = field(default=None)
    mse: float | None = field(default=None)


class Model(ABC):
    """
    Abstract base class for models that are used in Statsformer.
    """
    @abstractmethod
    def task(self) -> ModelTask:
        pass

    @abstractmethod
    def set_num_threads(self, num_threads):
        pass

    def using_sample_weights(self):
        return False

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior,
        random_seed: int=42
    ) -> "Model":
        pass

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        For classification, this will output logits
        """
        pass

    def logits_to_class(
        self, logits: np.ndarray
    ) -> np.ndarray:
        logits_to_class(logits, self.task())

    def predict_class(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Outputs integer or binary labels.
        Only for classification tasks
        """
        return self.logits_to_class(self.predict(X))
    
    def accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        return np.mean(
            self.predict_class(X) == y
        )

    def misclass(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        return 1 - self.accuracy(X, y)
    
    def auroc(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        # TODO: fix for multiclass
        return roc_auc_score(
            y, self.predict(X)
        )
    
    def mcc(
        self,
        X: np.ndarray,
        y: np.ndarray
    ):
        return matthews_corrcoef(
            y, self.predict_class(X)
        )

    def MSE(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        if self.task().value != ModelTask.REGRESSION.value:
            raise NotImplementedError(
                "MSE loss only available for regression"
            )
        return np.mean(
            (self.predict(X) - y) ** 2
        )
    
    def eval(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> EvalResult:
        """
        Runs all relevant evaluation metrics for the model's task
        and returns them in an EvalResult object.
        """
        y = y.reshape(-1)
        return get_metrics(y, self.predict(X), self.task())


def logits_to_class(
    logits: np.ndarray, task: ModelTask
):
    if task.value == ModelTask.BINARY_CLASSIFICATION.value:
        proba = expit(logits)
        val_probs = np.stack((1 - proba, proba), axis=-1).squeeze()
        return np.argmax(val_probs, axis=1)
    if task.value == ModelTask.MULTICLASS.value:
        return np.argmax(logits, axis=1)
    raise NotImplementedError(
        f"predict_class not applicable for task {task}"
    )


def get_metrics(
    y: np.ndarray, preds: np.ndarray, task: ModelTask
) -> EvalResult:
    if task.is_classification():
        classes = logits_to_class(preds, task)
        acc = np.mean(
            classes == y
        )
        return EvalResult(
            accuracy=acc,
            misclass=1 - acc,
            mcc=matthews_corrcoef(
                y, classes
            ),
            auroc=roc_auc_score(
                y, preds
            )
        )
    return EvalResult(
        mse=np.mean((y - preds)**2)
    )


class CVMetrics(Enum):
    ACC = "accuracy"
    AUROC = "roc_auc"
    MCC = "mcc"
    LOSS = "loss"

    def get_error_metric(
        self, y: np.ndarray,
        preds: np.ndarray,
        task: ModelTask,
    ) -> float:
        metrics = get_metrics(
            y, preds, task
        )
        if not task.is_classification():
            return metrics.mse
        if self.value == CVMetrics.ACC.value:
            return -metrics.accuracy
        if self.value == CVMetrics.AUROC.value:
            return -metrics.auroc
        if self.value == CVMetrics.MCC.value:
            return -metrics.mcc
        raise NotImplementedError(f"Metric {self.value} not implemented")

    def get_string(self):
        if self.value == CVMetrics.MCC.value:
            return "balanced_accuracy" # sklearn doesn't have MCC in CV, so we use
                                       # balanced_accuracy as a proxy
        return self.value


@dataclass
class CVOutput:
    """
    Returns a trained cross-validated model along with out-of-fold predictions
    (logits for classification, or raw predictions for regression) over the
    training data.
    """
    model: "ModelCV"
    oof_predictions: np.ndarray


class ModelCV(Model):
    """
    For models that are not internally cross-validated, this is a wrapper
    around the regular fit/prediction functions. For models with internal
    cross-validation, this allows one cross-validator to both:
    - train the model
    - provide out-of-fold predictions,
    which is far more efficient than running nested cross-validation loops.

    For models that are internally cross-validated, this class should be
    extended and the fit_cv function overridden as needed.
    """

    def _get_cross_validator(
        self, cv: int | BaseCrossValidator,
        y: np.ndarray, seed: int,
    ):
        return get_cross_validator(
            cv=cv,
            y=y,
            is_classification=self.task().is_classification(),
            seed=seed,
        )

    def fit_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior,
        cv: int | BaseCrossValidator=5,
        random_seed: int=42
    ):
        cv = self._get_cross_validator(cv, y, random_seed)
        
        if self.task().value == ModelTask.MULTICLASS.value:
            Z = np.zeros((X.shape[0], len(np.unique(y))))
        else:
            Z = np.zeros((X.shape[0]))

        # OOF predictions
        for (train_index, test_index) in cv.split(X, y):
            self.fit(
                X[train_index, :],
                y[train_index],
                feature_prior,
                random_seed=random_seed
            )
            Z[test_index] = self.predict(
                X[test_index, :]
            )
        # fit on all of the data
        self.fit(X, y, feature_prior)
        
        return CVOutput(self, oof_predictions=Z)