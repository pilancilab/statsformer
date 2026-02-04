import time
import numpy as np
import scipy
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from statsformer.models.base import CVMetrics, CVOutput, Model, ModelCV, ModelTask
from adelie.solver import grpnet
import adelie as ad
from adelie.diagnostic import predict, coefficient


from statsformer.prior import FeaturePrior


class Lasso(ModelCV):
    def __init__(
        self,
        task: ModelTask,
        cv_metric: CVMetrics=CVMetrics.LOSS,
        elasticnet_alpha: float=1,
        default_folds_cv: int=5,
        lambda_path_size: int=100,
        lambda_min_ratio: int=1e-2,
        num_threads: int=8,
        oversample_cv: bool=False,
    ):
        self.model_task = task
        self.cv_metric = cv_metric
        self.default_folds_cv = default_folds_cv
        self.lambda_path_size = lambda_path_size
        self.num_threads = num_threads
        self.lambda_min_ratio = lambda_min_ratio
        self.alpha = elasticnet_alpha
        self.oversample_cv = oversample_cv
    
        self.betas = None
        self.intercepts = None
        self.scaler = None
    
    def task(self) -> ModelTask:
        return self.model_task
    
    def set_num_threads(self, num_threads):
        self.num_threads = num_threads
    
    def _get_adelie_glm(
        self, y: np.ndarray
    ):
        y = np.asfortranarray(y.reshape(-1))
        if self.model_task == ModelTask.MULTICLASS:
            one_hot_encoder = OneHotEncoder(sparse_output=False)  # Use dense array
            y_one_hot = one_hot_encoder.fit_transform(y)
            return ad.glm.multinomial(y=y_one_hot, dtype=np.float32)
        if self.model_task == ModelTask.BINARY_CLASSIFICATION:
            return ad.glm.binomial(y=y, dtype=np.float32)
        # Otherwise, regression
        return ad.glm.gaussian(y=y, dtype=np.float32)

    def fit_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior,
        cv: int | BaseCrossValidator=5,
        random_seed: int=42
    ) -> "Lasso":
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X, copy=True)

        penalty = feature_prior.get_penalty()
        penalty /= np.sum(penalty)
        
        cv = self._get_cross_validator(
            cv, y, random_seed,
        )
            
        splits = list(cv.split(X, y))
        X = np.asfortranarray(X.astype(np.float32))
        y = y.reshape(-1)
        glm_full = self._get_adelie_glm(y)

        grpnet_kwargs=dict(
            ddev_tol=0,
            early_exit=False,
            n_threads=self.num_threads,
            progress_bar=False,
            alpha=self.alpha,
            penalty=penalty,
            tol=1e-5
        )

        ### The following code is adapted from adelie cv_grpnet
        # Fit once with all the data to get the lambdas
        state = grpnet(
            X=X,
            glm=glm_full,
            lmda_path_size=0,
            **grpnet_kwargs
        )

        full_lmdas = state.lmda_max * np.logspace(
            0, np.log10(self.lambda_min_ratio), self.lambda_path_size
        )

        cv_metrics = np.zeros((self.default_folds_cv, full_lmdas.shape[0]))

        if self.task().value == ModelTask.MULTICLASS.value:
            oof_predictions = np.zeros(
                (full_lmdas.shape[0], X.shape[0], len(np.unique(y)))
            )
        else:
            oof_predictions = np.zeros((full_lmdas.shape[0], X.shape[0]))
        
        for (fold, split) in enumerate(splits):
            train_index, test_index = split
            glm_train = self._get_adelie_glm(y[train_index])
            glm_test = self._get_adelie_glm(y[test_index])
            X_train = np.asfortranarray(X[train_index, :])

            # compute current lambda path augmented with full path
            # (required for convergence)
            state = grpnet(
                X=X_train,
                glm=glm_train,
                lmda_path_size=0,
                **grpnet_kwargs
            )

            curr_lmdas = state.lmda_max * np.logspace(
                0, np.log10(self.lambda_min_ratio), self.lambda_path_size
            )
            curr_lmdas = curr_lmdas[curr_lmdas > full_lmdas[0]]
            aug_lmdas = np.sort(np.concatenate([full_lmdas, curr_lmdas]))[::-1]

            state = grpnet(
                X=X_train,
                glm=glm_train,
                lmda_path=aug_lmdas,
                **grpnet_kwargs
            )

            # get coefficients/intercepts only on full_lmdas
            betas = state.betas
            intercepts = state.intercepts
            lmdas = state.lmdas
            if len(lmdas) == 0: # this fold failed completely so we just ignore it
                continue

            valid_lmda_idxs = [
                i for i in range(len(full_lmdas)) if \
                    full_lmdas[i] <= lmdas[0] and full_lmdas[i] > lmdas[-1]
            ]

            beta_ints = [
                coefficient(
                    lmda=lmda,
                    betas=betas,
                    intercepts=intercepts,
                    lmdas=lmdas,
                )
                for lmda in full_lmdas[valid_lmda_idxs]
            ]
            full_betas = scipy.sparse.vstack([x[0] for x in beta_ints])
            full_intercepts = np.array([x[1] for x in beta_ints])

            # compute linear predictions
            etas = predict(
                X=np.asfortranarray(X[test_index]),
                betas=full_betas,
                intercepts=full_intercepts,
                n_threads=self.num_threads,
            )

            cv_metrics[fold] = np.inf

            if self.cv_metric.value == CVMetrics.LOSS.value:
                cv_metrics[fold, valid_lmda_idxs] = np.array([
                    glm_test.loss(eta) for eta in etas
                ]) / len(test_index)
            else:
                cv_metrics[fold, valid_lmda_idxs] = np.array([
                    self.cv_metric.get_error_metric(y[test_index], eta, self.task()) for eta in etas
                ]) / len(test_index)

            if self.task().is_classification():
                oof_predictions[np.ix_(valid_lmda_idxs, test_index)] = etas

        # Sometimes some folds fail
        oof_predictions[np.abs(oof_predictions) > 1e4] = 0
        avg_losses = np.mean(cv_metrics, axis=0)
        best_idx = np.argmin(avg_losses)

        # retrain
        model = grpnet(
            X=X,
            glm=glm_full,
            lmda_path=full_lmdas[:best_idx+1],
            **grpnet_kwargs
        )
        if len(model.lmdas) == 0:
            model = grpnet(
                X=X,
                glm=glm_full,
                lmda_path=[full_lmdas[0]],
                **grpnet_kwargs
            )

        self.betas = model.betas[-1]
        self.intercepts = model.intercepts[-1]

        return CVOutput(
            model=self,
            oof_predictions=oof_predictions[best_idx]
        )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior,
        random_seed: int=42,
    ):
        self.fit_cv(
            X, y, feature_prior,
            cv=self.default_folds_cv,
            random_seed=random_seed
        )
        return self
    
    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        X = self.scaler.transform(X, copy=True)
        X = np.asfortranarray(X.astype(np.float32))
        res = predict(
            X=X,
            betas=self.betas,
            intercepts=self.intercepts,
            n_threads=self.num_threads,
        )
        if res.shape[0] == 1:
            res = res.reshape(-1)
        if self.task().is_classification():
            res[np.abs(res) > 1e4] = 0
        return res
    