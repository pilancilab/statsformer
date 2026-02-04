# Statsformer 🚀

We introduce *Statsformer*, a principled framework for integrating large language model (LLM)-derived knowledge into supervised statistical learning.
Existing approaches are limited in adaptability and scope: they either inject LLM guidance as an unvalidated heuristic, which is sensitive to LLM hallucination, or embed semantic information within a single fixed learner.
Statsformer overcomes both limitations through a guardrailed ensemble architecture.
We embed LLM-derived feature priors within an ensemble of linear and nonlinear learners, adaptively calibrating their influence via cross-validation.
This design yields a flexible system with an oracle-style guarantee that it performs no worse than any convex combination of its in-library base learners, up to statistical error.
Empirically, informative priors yield consistent performance improvements, while uninformative or misspecified LLM guidance is automatically downweighted, mitigating the impact of hallucinations across a diverse range of prediction tasks.

🔗 Paper link: [*Statsformer*: Validated Ensemble Learning with LLM-Derived Semantic Priors.](https://arxiv.org/pdf/2601.21410)

## Setup
1. Make a Python environment with Python 3.11 (some dependencies don't have wheels for newer versions)
2. If needed (e.g., for a conda environment), install `pip`.
3. Run `pip install --editable . `
4. Copy `.env.sample` to `.env` and fill in API keys.

## Experiment Pipeline
In general, experiments are run in the following order:
1. **Dataset building**: e.g., `bash ./shell_scripts/datasets/build_bank_marketing.sh`. This saves the dataset in a specified format within `data/datasets`
2. **Score generation**: e.g., `bash ./shell_scripts/llm/generate_scores_bank_marketing.sh`. This generates scores for each feature and saves them in `data/generated_priors`
3. **Tuning experiments**: looking at how our method works on a single class of models, e.g., `bash shell_scripts/stats/run_all_single_learner.sh` will run this for all models and datasets. Or you can run `python scripts/tuning/single_learner_study.py --learner $METHOD --dataset $DATASET` for a single learner and dataset.
4. **Final experiments: classification or regression vs. baselines**: e.g., `bash shell_scripts/stats/run_all_baseline_comp.sh`. This runs baseline methods and `statsformer` on the given classification or regression task.

See `shell_scripts` for more scripts.

_Note_: To run the AutoML-Agent baseline, you have to run `git submodule update --init --recursive` to clone the AutoML-Agent repository into `automl-agent`.
Then, set up a Python environment with Python 3.11 and install the dependencies in `automl-agent/requirements.txt`.
You may also need to set up API keys as environment variables; see `automl-agent/configs.py` for details.

## FAQ:
- If you see the error `UserWarning: Wrong extension .rds for file in RDATA format` when building the ETP dataset, this is expected and can be safely ignored.

## Repository Structure

### `prompts`
Contains prompt templates used for interacting with large language models.

- `default_prompt.txt`: The default prompt template for collecting feature scores from the LLM.
- `default_system_prompt.txt`: The default system prompt
- `task_descriptions\<task_name.json>`: Task-specific descriptions to guide the LLM for different datasets. Each JSON file contains two fields:
    - `context`: background information about the task.
    - `task`: the specific classification or regression task description, e.g., "classifying follicular lymphoma vs. diffuse large B-cell lymphoma".
- `rag/default_retriever.txt`: The default prompt template for retrieval-augmented generation (RAG) setups.

### `scripts`
Contains Python scripts to run the main experiment components:
- `datasets/`: dataset building: collects data, targets, and feature names, and saves then in a standardized format.
- `llm/`: generates feature scores (i.e., feature priors) for a dataset
- `stats/`: evaluation of statsformer on classification/regression tasks, compared to baseline methods.

### `shell_scripts`
Bash scripts for running experiments in `scripts`.

### `src/statsformer`
Main python package for `statsformer`, covering dataset formation, LLM score collection, baseline methods, the statistical components of our method, and adversarial simulations.

**A. Datasets (`data/`):**
- `data/dataset.py`: contains the `Dataset` class, which handles train/test splits, loading/saving datasets, and basic dataset operations (e.g., loading data for a single train/test split).
    - _Splitting_: To evaluate our method, we set a fixed test ratio (e.g., 0.2), and sweep the training ratio from (1 - test_ratio) down to a small value (e.g., 0.1). For each training ratio, we generate multiple random train/test splits to ensure robust evaluation.
    - _Loading/Saving_: Datasets are saved in a directory with `X.csv` (features) and `y.csv` (targets). The `splits/` directory within the dataset directory saves the indices of all train and test splits as npy files. The `Dataset` class provides methods to load from and save to this format.
- `data/preselection.py`: contains the `DatasetWithPreselection` class, which extends `Dataset` to include feature preselection methods. For each train/test split, it preselects a specified number of features using a given data-driven method (e.g., mutual information, XGBoost importance) and saves the selected feature names.

The rest of the files in `statsformer.data` handle dataset downloading, preprocessing, and specific dataset builders. All datasets are standardized to the same format using the `Dataset` class.

**B. LLM Score Collection (`llm/`):**
- `llm/base.py`: logic for calling LLMs (e.g., through the OpenAI or OpenRouter APIs), handling retries, tracking cost, and parsing outputs as either JSON or Pydantic `BaseModel`.
    - The main class is `LLM`, which provides methods to send prompts and receive structured (or unstructured) outputs.
    - Outputs are wrapped in the `LLMOutput` class, which tracks metadata like success status, number of retries, raw outputs, and costs.
- `llm/common.py`: common utilities for LLM interactions, including client instantiation and configuration management.
- `llm/prompting.py`: prompt templates and utilities for filling in prompts. This includes templates for feature scoring with and without retrieval-augmented generation (RAG), as well as output format instructions.
- `llm/generated_prior.py`: main logic for generating feature scores using LLMs. This includes batching features, constructing prompts, sending requests to the LLM, parsing outputs, and saving the generated scores.
    - The main class is `GeneratedPrior`, which orchestrates the score generation process (collecting scores for all features, possibly with multiple trials), tracking cost and optionally model reasoning. In particular, the `generate` method handles the core logic of batching features, sending prompts, and parsing outputs.
    - The `load_scores` method (outside of the `GeneratedPrior` class) loads previously generated scores from disk for evaluation. `GeneratedPrior.generate` automatically saves scores to disk as they are generated.
- `llm/with_preselection.py`: extends `GeneratedPrior` to handle feature preselection. It uses a `DatasetWithPreselection` instance, which includes the preselected features for each train/test split. Scores are generated only for the preselected features. As the preselected features may differ across splits, scores are saved separately for each split.
- `llm/rag`: scaffolding for adding a RAG system (optional).

**C. Our Method (`models/`, `metalearning/`, `prior.py`):**
- `prior.py`: contains the `FeaturePrior` class, which transforms LLM-generated feature scores into feature importance scores, penalties, and sample weights according to a specified functional form and temperature.
    - The `FeaturePrior` class takes in raw feature scores and a configuration (`FeaturePriorConfig`) specifying the functional form (e.g., power-law) and temperature. It provides methods to compute feature importance scores (`get_score`), penalties (`get_penalty`), and sample weights (`get_sample_weights`).
    - `FeaturePrior` objects are passed downstream into the `fit` function statsformer's constituent models (e.g., Lasso), which use the feature scores for regularization and weighting.
- `models/`: contains the constituent models used for statsformer.
    - `base.py`: contains abstract classes that all constituent models must inherit from: `Model`, which has `fit` and `predict` methods, and `ModelCV`, which extends `Model` to include k-fold cross-validation logic (production of out-of-fold predictions).
    All constituent models for statsformer inherit from `ModelCV`.
    - Other files, e.g., `glm.py`, `kernel.py`, `xgboost.py`, implement specific constituent models.
- `metalearning/`: the out-of-fold stacking ensemble logic.
    - `base.py`: some common dataclasses; no actual code.
    - `stacking.py`: the main logic for performing out-of-fold (OOF) ensembling. The `OOFStacking` class implements the out-of-fold stacking ensemble. It takes in a list of constituent model configurations, performs out-of-fold predictions for each model, learns weights using a meta-learner (e.g., non-negative logistic regression for classification, elastic net for regression), and refits each base model on the full data with the learned weights.
        - The `fit` method orchestrates the fitting process, while the `predict` method makes predictions using the fitted ensemble.
    - `metalearenrs.py`: includes the non-negative logistic regression implementation.

**D. Baseline Methods (`baselines/`):**
- `baselines/data_driven.py`: contains data-driven baseline methods, including Lasso, Random Forest, XGBoost, and LassoNet. Each method extends the `Model` class from `models/base.py` and implements the `fit` and `predict` methods.
    - Several of these methods take logic directly from `statsformer.models`, e.g., the Lasso implementation uses `statsformer.models.glm.LassoCVModel`, passing in uniform feature scores.

**E. Experiment Pipelines (`experiment/`):**
- `experiment/base.py`: contains the `BaseExperiment` and `Experiment` classes, which handle running experiments on datasets, including fitting baseline methods and statsformer configurations, collecting results, and plotting performance.
    - The `BaseExperiment` class provides core logic for running experiments, including fitting methods on multiple train/test splits, collecting results, and plotting performance.
    - The `Experiment` class extends `BaseExperiment` and is the primary class used for running experiments on a dataset. It has methods `run_baselines` and `run_statsformer` to fit baseline methods and statsformer configurations, respectively.
- `experiment/base_learner_study.py`: contains the `BaseLearnerStudy` class, which evaluates the performance of an individual constituent model (base learner) used in statsformer, for different transformations performed on the LLM-generated feature scores (e.g., different functional forms and temperatures).

## Citation
If you find our work useful, consider giving our repo a ⭐ and citing our paper as:
<pre>@misc{zhang2026statsformervalidatedensemblelearning,
      title={Statsformer: Validated Ensemble Learning with LLM-Derived Semantic Priors}, 
      author={Erica Zhang and Naomi Sagan and Danny Tse and Fangzhao Zhang and Mert Pilanci and Jose Blanchet},
      year={2026},
      eprint={2601.21410},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2601.21410}}
</pre>

