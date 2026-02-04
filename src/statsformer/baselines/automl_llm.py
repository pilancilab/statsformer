import numpy as np
import pandas as pd
import random
import shutil
import tempfile
import sys
import os
import json
import pickle
from pathlib import Path

# Try to import joblib for model loading (generated code uses joblib.dump)
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False

# Add automl-agent directory to Python path
_project_root = Path(__file__).parent.parent.parent.parent
_automl_agent_dir = _project_root / "automl-agent"
if _automl_agent_dir.exists() and str(_automl_agent_dir) not in sys.path:
    sys.path.insert(0, str(_automl_agent_dir))

try:
    from agent_manager import AgentManager
    from operation_agent.execution import execute_script
    _automl_agent_available = True
except ImportError as e:
    AgentManager = None
    execute_script = None
    _automl_agent_import_error = e
    _automl_agent_available = False

from statsformer.llm.prompting import fill_in_prompt_file
from statsformer.models.base import Model, ModelTask
from statsformer.prior import FeaturePrior
from statsformer.utils import clipped_logit


class AutoMLAgentBaseline(Model):
    def __init__(
        self,
        task: ModelTask,
        time_limit: int = 60,
        automl_agent_path: str | None = None,
        llm: str = "gpt-4",
        n_plans: int = 1,
        n_candidates: int = 3,
        n_revise: int = 1,
        interactive: bool = False,
        full_pipeline: bool = False,
        device: int = 0,
        **automl_agent_kwargs
    ):
        if not _automl_agent_available:
            error_msg = (
                "AutoML Agent not found. Please ensure automl-agent directory is "
                "available in the project root and dependencies are installed."
            )
            if '_automl_agent_import_error' in globals():
                error_msg += f"\nOriginal error: {_automl_agent_import_error}"
            raise ImportError(error_msg)
        
        self.model_task = task
        self.time_limit = time_limit
        self.automl_agent_kwargs = automl_agent_kwargs
        self.automl_agent_path = automl_agent_path
        self.model = None
        self.preprocessor = None
        self.agent_manager = None
        self.num_threads = -1
        self.temp_dir = None
        self.data_dir = None
        self.workspace_dir = None
        
        self.llm = llm
        self.n_plans = n_plans
        self.n_candidates = n_candidates
        self.n_revise = n_revise
        self.interactive = interactive
        self.full_pipeline = full_pipeline
        self.device = device
    
    def task(self) -> ModelTask:
        return self.model_task
    
    def set_num_threads(self, num_threads):
        self.num_threads = num_threads
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, random_seed: int):
        if self.data_dir is None:
            self.data_dir = tempfile.mkdtemp(prefix="automl_data_")
        
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_df = pd.Series(y.ravel(), name="target")
        data = pd.concat([X_df, y_df], axis=1)
        
        data_path = Path(self.data_dir) / "data.csv"
        data.to_csv(data_path, index=False)
        
        return str(data_path.parent)
    
    def create_user_prompt(self, model_save_path: str = None) -> str:
        if self.task().is_classification():
            if self.task() == ModelTask.BINARY_CLASSIFICATION:
                task_desc = "binary classification"
            else:
                task_desc = "multiclass classification"
        else:
            task_desc = "regression"
        
        model_path = model_save_path if model_save_path else './agent_workspace/trained_models/model.pkl'
        return fill_in_prompt_file(
            "./prompts/automl_agent_prompt.txt",
            dict(
                task_desc=task_desc,
                model_path=model_path
            )
        )
    
    def extract_model_from_code(self, code_path: str, preferred_model_path: str = None):
        code_path_obj = Path(code_path) if code_path else None
        
        model_paths = []
        # Add preferred path first if provided
        if preferred_model_path:
            model_paths.append(Path(preferred_model_path))
        
        # Add standard paths
        model_paths.extend([
            Path("agent_workspace/trained_models/model.pkl"),
            Path("./agent_workspace/trained_models/model.pkl"),
        ])
        
        if code_path_obj:
            model_paths.extend([
                code_path_obj.parent / "trained_models" / "model.pkl",
                code_path_obj.parent.parent / "trained_models" / "model.pkl",
            ])
        
        if self.workspace_dir:
            model_paths.append(self.workspace_dir / "trained_models" / "model.pkl")
        
        if code_path_obj and code_path_obj.exists():
            model_paths.append(code_path_obj.parent / "model.pkl")
            code_dir = code_path_obj.parent
            if code_dir.exists():
                pkl_files = list(code_dir.glob("*.pkl"))
                if pkl_files:
                    model_paths.extend(pkl_files)
        
        for path in model_paths:
            if path and path.exists():
                # Try joblib first (generated code uses joblib.dump)
                if JOBLIB_AVAILABLE:
                    try:
                        loaded_obj = joblib.load(path)
                        # Handle case where loaded object is a dict with 'model' key
                        if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
                            model = loaded_obj['model']
                            self.preprocessor = loaded_obj.get('preprocessor', None)
                            if self.preprocessor is not None:
                                print(f"[AutoML Agent] Successfully loaded model and preprocessor from {path} using joblib")
                            else:
                                print(f"[AutoML Agent] Successfully loaded model from {path} using joblib (extracted from dict)")
                        else:
                            model = loaded_obj
                            print(f"[AutoML Agent] Successfully loaded model from {path} using joblib")
                        return model
                    except Exception as e:
                        print(f"[AutoML Agent] Failed to load with joblib: {e}. Trying pickle...")
                
                # Fallback to pickle
                try:
                    with open(path, 'rb') as f:
                        loaded_obj = pickle.load(f)
                    # Handle case where loaded object is a dict with 'model' key
                    if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
                        model = loaded_obj['model']
                        self.preprocessor = loaded_obj.get('preprocessor', None)
                        if self.preprocessor is not None:
                            print(f"[AutoML Agent] Successfully loaded model and preprocessor from {path} using pickle")
                        else:
                            print(f"[AutoML Agent] Successfully loaded model from {path} using pickle (extracted from dict)")
                    else:
                        model = loaded_obj
                        print(f"[AutoML Agent] Successfully loaded model from {path} using pickle")
                    return model
                except Exception as e:
                    print(f"[AutoML Agent] Failed to load with pickle: {e}. Trying next path...")
                    continue
        
        return None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_prior: FeaturePrior=None,
        random_seed: int=42,
    ) -> "AutoMLAgentBaseline":
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        data_dir = self.prepare_data(X, y, random_seed)
        
        if self.automl_agent_path is None:
            self.temp_dir = tempfile.mkdtemp(prefix="automl_agent_")
            workspace_base = self.temp_dir
        else:
            workspace_base = self.automl_agent_path
        
        if self.task().is_classification():
            task_type = "tabular_classification"
        else:
            task_type = "tabular_regression"
        
        import time
        uid = f"statsformer_{int(time.time())}_{random_seed}"
        
        # Create unique model save path for this trial
        self.model_save_path = f"./agent_workspace/trained_models/model_{uid}.pkl"
        self.model_save_dir = Path("./agent_workspace/trained_models")
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create prompt with unique model path
        user_prompt = self.create_user_prompt(self.model_save_path)
        
        self.agent_manager = AgentManager(
            task=task_type,
            n_plans=self.n_plans,
            n_candidates=self.n_candidates,
            n_revise=self.n_revise,
            device=self.device,
            interactive=self.interactive,
            llm=self.llm,
            data_path=data_dir,
            full_pipeline=self.full_pipeline,
            rap=True,
            decomp=True,
            verification=True,
            uid=uid,
            **self.automl_agent_kwargs
        )
        
        print(f"[AutoML Agent] Starting pipeline for {task_type} task...")
        print(f"[AutoML Agent] Model will be saved to: {self.model_save_path}")
        self.agent_manager.initiate_chat(user_prompt)
        
        code_path = f"./agent_workspace/exp{self.agent_manager.code_path}.py"
        self.workspace_dir = Path("./agent_workspace/exp").resolve()
        
        self.model = self.extract_model_from_code(code_path, self.model_save_path)
        
        if self.model is None:
            raise ValueError(
                f"Could not load trained model from generated code. "
                f"Expected model at various locations relative to {code_path}. "
                f"Please check that the AutoML Agent generated code saved the model properly."
            )
        
        return self
    
    def cleanup(self):
        # Use getattr with defaults to handle cases where attributes might not be initialized
        temp_dir = getattr(self, 'temp_dir', None)
        data_dir = getattr(self, 'data_dir', None)
        
        for dir_path in [temp_dir, data_dir]:
            if dir_path is not None and os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    # Silently ignore cleanup errors
                    pass
        
        if hasattr(self, 'temp_dir'):
            self.temp_dir = None
        if hasattr(self, 'data_dir'):
            self.data_dir = None
    
    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            # Silently ignore errors during cleanup
            pass
    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        # Apply preprocessor if available
        if self.preprocessor is not None:
            try:
                X_transformed = self.preprocessor.transform(X_df)
                # Handle sparse matrices and numpy arrays
                if hasattr(X_transformed, 'toarray'):
                    X_transformed = X_transformed.toarray()
                # Convert to DataFrame if needed (preprocessor might change number of columns)
                if isinstance(X_transformed, np.ndarray):
                    X_df = pd.DataFrame(X_transformed)
                else:
                    X_df = pd.DataFrame(X_transformed)
            except Exception as e:
                print(f"[AutoML Agent] ERROR: Preprocessor transform failed: {e}. Exiting.")
                raise e
        
        if hasattr(self.model, 'predict_proba') and self.task().is_classification():
            proba = self.model.predict_proba(X_df)
            if self.task() == ModelTask.BINARY_CLASSIFICATION:
                if isinstance(proba, pd.DataFrame):
                    pos_class = proba.iloc[:, -1].values
                else:
                    pos_class = proba[:, -1] if proba.ndim > 1 else proba
                return clipped_logit(pos_class)
            else:
                return proba
        elif (not self.task().is_classification()) and hasattr(self.model, 'predict'):
            predictions = self.model.predict(X_df)
            if isinstance(predictions, pd.Series):
                return predictions.values
            return np.array(predictions).ravel()
        else:
            raise ValueError(
                f"Loaded model does not have 'predict' or 'predict_proba' methods. "
                f"Model type: {type(self.model)}"
            )
