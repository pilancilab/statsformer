import cvxpy as cvx
import numpy as np
from dataclasses import dataclass

from scipy.optimize import minimize
from sklearn._loss.loss import HalfBinomialLoss
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LogisticRegressionCV
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.model_selection import BaseCrossValidator
from statsformer.models.base import CVMetrics, ModelTask
from enum import Enum

from statsformer.utils import clipped_logit, get_cross_validator


class MetaLearnerType(Enum):
    DEFAULT = "default"
    SIMPLEX = "simplex"
    NONEG = "nonnegative"

    def build(
        self,
        cv_metric: CVMetrics,
        num_threads: int=16,
        fit_intercept: bool=False,
        oversample_cv: bool=False,
    ) -> "MetaLearner":
        kwargs = dict(
            cv_metric=cv_metric,
            num_threads=num_threads,
            fit_intercept=fit_intercept,
            oversample_cv=oversample_cv
        )
        if self.value == MetaLearnerType.DEFAULT.value:
            return MetaLearner(**kwargs)
        if self.value == MetaLearnerType.SIMPLEX.value:
            return SimplexMetaLearner(**kwargs)
        if self.value == MetaLearnerType.NONEG.value:
            return PositiveMetaLearner(**kwargs)
        raise NotImplementedError(
            f"Meta-learner type {self.value} not yet implemented or unknown"
        )


@dataclass
class MetaLearnerOutput:
    weights: np.ndarray
    intercept: np.ndarray | float

    def get_weight_for_method(self, i):
        if len(self.weights.shape) > 1 and self.weights.shape[1] > 1:
            return self.weights[i, :]
        else:
            return self.weights[i]

class MetaLearner:
    def __init__(
        self, num_threads,
        cv_metric: CVMetrics,
        fit_intercept: bool=False,
        oversample_cv: bool=False,
    ):
        self._num_threads = num_threads
        self.fit_intercept = fit_intercept
        self._cv_metric = cv_metric
        self.oversample_cv = oversample_cv

    @property
    def num_threads(self) -> int:
        return self._num_threads

    @property
    def cv_metric(self) -> CVMetrics:
        return self._cv_metric

    def set_num_threads(self, num_threads):
        self._num_threads = num_threads

    def fit_binary(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        cv: int | BaseCrossValidator,
        random_seed: int=42,
    ) -> MetaLearnerOutput:
        meta_learner = LogisticRegressionCV(
            cv=get_cross_validator(
                cv=cv,
                y=y,
                is_classification=True,
                oversample=self.oversample_cv,
                seed=random_seed
            ),
            scoring=self.cv_metric.get_string(),
            n_jobs=self.num_threads,
            fit_intercept=self.fit_intercept,
            random_state=random_seed
        ).fit(Z, y)
        return MetaLearnerOutput(
            weights=meta_learner.coef_[0],
            intercept=meta_learner.intercept_
        )

    def fit_multiclass(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        cv: int | BaseCrossValidator,
        random_seed: int=42,
    ) -> MetaLearnerOutput:
        classes = np.unique(y)
        intercept = np.zeros(len(classes))
        pi = np.ones(Z.shape[1], len(classes)) / Z.shape[1]

        for (j, cls) in enumerate(classes):
            out = self.fit_binary(Z[:, j, :], (y == cls), cv=cv, random_seed=random_seed)
            intercept[j] = out.intercept
            pi[:, j] = out.weights
        return MetaLearnerOutput(
            weights=pi,
            intercept=intercept
        )

    def fit_regression(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        cv: int | BaseCrossValidator,
        random_seed: int=42,
        _positive: bool=False
    ) -> MetaLearnerOutput:
        meta_learner = ElasticNetCV(
            cv=get_cross_validator(
                cv=cv,
                y=y,
                is_classification=False,
                seed=random_seed
            ),
            n_jobs=self.num_threads,
            fit_intercept=self.fit_intercept,
            positive=_positive,
            random_state=random_seed
        ).fit(Z, y)
        return MetaLearnerOutput(
            weights=meta_learner.coef_,
            intercept=meta_learner.intercept_
        )

    def fit(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        cv: int | BaseCrossValidator,
        model_task: ModelTask,
        random_seed: int=42,
    ) -> MetaLearnerOutput:
        if model_task.value == ModelTask.BINARY_CLASSIFICATION.value:
            return self.fit_binary(Z, y, cv, random_seed=random_seed)
        if model_task.value == ModelTask.MULTICLASS.value:
            return self.fit_multiclass(Z, y, cv, random_seed=random_seed)
        if model_task.value == ModelTask.REGRESSION.value:
            return self.fit_regression(Z, y, cv, random_seed=random_seed)
        raise NotImplementedError(
            f"Unknown model task {model_task.value}"
        )


class PositiveMetaLearner(MetaLearner):
    def __init__(
        self, num_threads,
        cv_metric: CVMetrics,
        fit_intercept: bool=False,
        oversample_cv: bool=False,
        Cs: int=10,
        max_iter: int=100,
        tol: float=1e-3,
    ):
        super().__init__(
            num_threads=num_threads,
            cv_metric=cv_metric,
            fit_intercept=fit_intercept,
            oversample_cv=oversample_cv
        )
        self.Cs = Cs
        self.max_iter = max_iter
        self.tol = tol

    def fit_regression(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        cv: int | BaseCrossValidator,
        random_seed: int=42,
    ) -> MetaLearnerOutput:
        return super().fit_regression(
            Z, y, cv, _positive=True, random_seed=random_seed
        )

    def _log_reg_path(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        Cs: np.ndarray,
        coef_init: np.ndarray | None=None,
    ):
        """
        Adapted from sklearn's LogisticRegressionCV
        """
        n_samples, n_features = Z.shape

        y = y.astype(Z.dtype)

        w0 = np.ones(n_features + int(self.fit_intercept), dtype=Z.dtype) / n_features
        if self.fit_intercept:
            w0[-1] = 0
        if coef_init is not None:
            w0 = coef_init
        sw_sum = n_samples

        loss = LinearModelLoss(
            base_loss=HalfBinomialLoss(), fit_intercept=self.fit_intercept
        )
        func = loss.loss_gradient

        coefs = list()
        for _, C in enumerate(Cs):
            l2_reg_strength = 1.0 / (C * sw_sum)
            opt_res = minimize(
                func,
                w0,
                method="L-BFGS-B",
                jac=True,
                args=(Z, y, None, l2_reg_strength, self.num_threads),
                options={
                    "maxiter": self.max_iter,
                },
                bounds=[
                    (0, np.inf) for _ in range(n_features)
                ]
            )
            w0, loss = opt_res.x, opt_res.fun
            coefs.append(w0.copy())
        return coefs

    def _scoring_path(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        train: list[int],
        test: list[int],
        Cs: np.ndarray
    ):
        """
        Adapted from sklearn's LogisticRegressionCV
        """

        Z_train = Z[train]
        Z_test = Z[test]
        y_train = y[train]
        y_test = y[test]
        coefs = self._log_reg_path(
            Z_train, y_train, Cs
        )

        log_reg = LogisticRegression(solver="L-BFGS-B")
        log_reg.classes_ = np.unique(y)
        scores = list()

        for w in coefs:
            if self.fit_intercept:
                log_reg.coef_ = w[..., :-1]
                log_reg.intercept_ = w[..., -1]
            else:
                log_reg.coef_ = w
                log_reg.intercept_ = 0.0
            logits = clipped_logit(log_reg.predict_proba(Z_test)[:, 1])
            scores.append(-self.cv_metric.get_error_metric(y_test, logits, ModelTask.BINARY_CLASSIFICATION))

        return coefs, np.array(scores)

    def fit_binary(self, Z, y, cv, random_seed: int=42):
        """
        Adapted from sklearn's LogisticRegressionCV
        """

        # folds
        cv = get_cross_validator(
            cv=cv,
            y=y,
            is_classification=True,
            oversample=self.oversample_cv,
            seed=random_seed
        )
        folds = list(cv.split(Z, y))
        Cs = np.logspace(-4, 4, self.Cs)

        fold_coefs_ = [
            self._scoring_path(Z, y, train, test, Cs=Cs) \
                for train, test in folds
        ]
        coefs_paths, scores = zip(*fold_coefs_)
        coefs_paths = np.reshape(
            coefs_paths, (len(folds), len(Cs), -1)
        )
        scores = np.reshape(scores, (len(folds), len(Cs)))

        scores_sum = scores.sum(axis=0)
        best_index = np.unravel_index(np.argmax(scores_sum), scores_sum.shape)
        C = Cs[best_index[0]]
        coef_init = coefs_paths[0, *best_index, :]

        w = self._log_reg_path(Z, y, Cs=[C], coef_init=coef_init)[0]

        coef = w[:Z.shape[1]]
        if self.fit_intercept:
            intercept = w[-1]
        else:
            intercept = 0

        return MetaLearnerOutput(
            weights=coef,
            intercept=intercept
        )


class SimplexMetaLearner(PositiveMetaLearner):
    def _log_reg_path(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        Cs: np.ndarray,
        coef_init: np.ndarray | None=None
    ):
        """
        Adapted from sklearn's LogisticRegressionCV
        """
        n_samples, n_features = Z.shape
        y = y.astype(Z.dtype)

        w0 = np.ones(n_features + int(self.fit_intercept), dtype=Z.dtype) / n_features
        if self.fit_intercept:
            w0[-1] = 0
        if coef_init is not None:
            w0 = coef_init
        sw_sum = n_samples

        coefs = list()
        for _, C in enumerate(Cs):
            w = cvx.Variable(w0.shape)
            w.value = w0
            l2_reg_strength = 1.0 / (C * sw_sum)

            loss = cvx.sum(cvx.logistic(-y.T @ (Z @ w))) / len(y)
            reg = l2_reg_strength * cvx.norm(w, 2)

            constraints = [
                w >= 0,
                cvx.sum(w) == 1
            ]

            problem = cvx.Problem(cvx.Minimize(loss + reg), constraints)
            problem.solve(solver=cvx.ECOS)

            w0 = w.value
            coefs.append(w0.copy())
        return coefs
