from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from typing import Any, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

from pydantic import BaseModel, create_model

from statsformer.llm.common import LLMConfig
from statsformer.llm.base import LLM, LLMOutput
from statsformer.llm.pricing.base import LLMCost
from statsformer.llm.prompting import (
    OUTPUT_FORMAT_NO_REASONING, OUTPUT_FORMAT_REASONING, PROMPT_WITH_RAG,
    PROMPT_WITHOUT_RAG, PriorGenerationPrompt, fill_in_prompt
)
from statsformer.llm.rag.base import RAGConfig, RAGTypes
from statsformer.utils import dataclass_to_json


MAX_BATCH_SIZE = 100


# Output formats for LLM responses, with and without reasoning
class OutputModel(BaseModel):
    scores: dict[str, float]


class OutputModelReasoning(BaseModel):
    scores: dict[str, float]
    reasoning: dict[str, str]


@dataclass
class GeneratedPriorMetadata:
    """
    All configuration information for a score generation experiment.
    """
    prompt_config: dict
    llm_config: dict
    batch_size: int

    trial_dir: str
    failed_trial_dir: str
    scores_csv: str
    avg_scores_filename: str
    metadata_filename: str
    cost_filename: str

    feature_names_as_json: str

    rag_config: dict = field(default_factory=lambda: asdict(RAGConfig.disabled()))
    completed_trials: int = field(default=0)
    trial_filenames: list[str] = field(default_factory=list)

    def get_feature_names(self):
        return json.loads(self.feature_names_as_json)

    @staticmethod
    def _filename(dir: str | Path):
        return Path(dir) / "metadata.json"

    @classmethod
    def from_dir(cls, dir: str | Path):
        """
        Loads GeneratedPriorMetadata from disk, if it exists. Returns None if not
        found.
        """
        path = cls._filename(dir)
        if path.exists():
            return cls(
                **json.loads(path.read_text())
            )
        return None
    
    def save(self):
        return dataclass_to_json(self, self.metadata_filename)

    @classmethod
    def build(
        cls,
        prompt_config: PriorGenerationPrompt,
        llm_config: LLMConfig,
        rag_config: RAGConfig,
        batch_size: int,
        output_dir: str | Path,
        feature_names: list[str],
    ) -> "GeneratedPriorMetadata":
        """
        Given a list of configuration options, builds a GeneratedPriorMetadata
        object and creates the necessary output directories on disk.
        """
        output_dir = Path(output_dir)
        trial_dir = output_dir / "trials"
        failed_trial_dir = output_dir / "failed_trials"

        Path(trial_dir).mkdir(exist_ok=True, parents=True)
        Path(failed_trial_dir).mkdir(exist_ok=True, parents=True)

        return cls(
                metadata_filename=str(cls._filename(output_dir)),
                prompt_config=asdict(prompt_config),
                llm_config=asdict(llm_config),
                rag_config=asdict(rag_config),
                batch_size=batch_size,

                # filenames
                cost_filename=str(output_dir / "total_cost.json"),
                trial_dir=str(trial_dir),
                failed_trial_dir=str(failed_trial_dir),
                scores_csv=str(output_dir / "scores.csv"),
                avg_scores_filename=str(output_dir / "avg_scores.npy"),
                feature_names_as_json=json.dumps(feature_names),
            )

    def add_trial(self, trial_filename):
        """
        Called upon completion of a trial to update metadata.
        """
        self.completed_trials += 1
        self.trial_filenames.append(str(trial_filename))
        self.save()
    
    def print(self):
        print(("=" * 20) + "\nGeneratedPriorMetadata\n" + ("=" * 20))
        print(f"prompt_config={json.dumps(self.prompt_config, indent=2)}")
        print(f"llm_config={json.dumps(self.llm_config, indent=2)}")
        if self.rag_config["rag_type"] != RAGTypes.DISABLED.value:
            print(f"rag_config={json.dumps(self.rag_config, indent=2)}")
        else:
            print(f"rag_config=DISABLED")
        print(f"batch_size={self.batch_size}")
        print(f"feature_names={self.feature_names_as_json}")
        print("=" * 20)


@dataclass
class Trial:
    """
    Results for one round of collection scored for all features. One experiment
    can have multiple trials, which are averaged together to produce the final
    scores.
    """
    scores: dict[str, float] = field(default_factory=dict)
    reasoning: dict[str, str] = field(default_factory=dict)
    success: bool = field(default=True)
    total_cost: LLMCost = field(default_factory=LLMCost)
    raw_outputs: list[LLMOutput] = field(default_factory=list)

    def __add__(self, other: "Trial"):
        """
        For adding results for two batches of features together.
        """
        return Trial(
            scores={**self.scores, **other.scores},
            reasoning={**self.reasoning, **other.reasoning},
            success=self.success and other.success,
            total_cost=self.total_cost + other.total_cost,
            raw_outputs=self.raw_outputs + other.raw_outputs
        )

    def save(self, filename: str):
        """
        Saves the trial results to disk.
        """
        return dataclass_to_json(self, filename)


class GeneratedPrior:
    """
    Main logic for generating LLM scores for features.
    
    Supports multiple trials and RAG, and saves all final and intermediate
    results to disk, allowing the user to continue a failed experiment by
    loading past results from disk.
    """
    def __init__(
        self,
        feature_names: list[str],
        output_dir: str | Path,
        prompt_config: PriorGenerationPrompt,
        llm_config: LLMConfig,
        rag_config: RAGConfig,
        batch_size: int=MAX_BATCH_SIZE, # Number of features to score per LLM call
        clear: bool=False, # Whether to clear the output directory if it exists
        _expect_existing_metadata: bool=False # For internal use; whether to expect
                                              # existing metadata in the output dir
    ):
        # Possibly clear the output directory
        self.output_dir = Path(output_dir)
        if clear and self.output_dir.exists():
                print(f"[WARNING] Clearing non-empty directory {self.output_dir}")
                shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_names = feature_names

        # See if there is already existing metadata for this experiment. If so,
        # this overrides anything the user passed in.
        self.metadata = GeneratedPriorMetadata.from_dir(output_dir)
        if self.metadata is None:
            self.metadata = GeneratedPriorMetadata.build(
                prompt_config=prompt_config,
                llm_config=llm_config,
                rag_config=rag_config,
                batch_size=batch_size,
                output_dir=output_dir,
                feature_names=feature_names
            )
            self.feature_names = self.metadata.get_feature_names()
            rag_config = RAGConfig(**self.metadata.rag_config)
        elif not _expect_existing_metadata:
            print(f"[INFO] Found existing metadata file. Ignoring configuration arguments.")
            print(f"[INFO] Running score collection with the following configuration:")
            self.metadata.print()
        self.metadata.save()
        self.rag_config = rag_config
        
        # cost tracking
        self.cost_filename = Path(self.metadata.cost_filename)
        if self.cost_filename.exists():
            self.cost = LLMCost(
                **json.loads(self.cost_filename.read_text())
            )
        else:
            self.cost = LLMCost()
    
    @classmethod
    def from_dir(
        cls,
        dir: str | Path,
    ):
        """
        Loads a GeneratedPrior experiment from disk.
        """
        metadata = GeneratedPriorMetadata.from_dir(dir)
        return cls(
            feature_names=metadata.get_feature_names(),
            output_dir=dir,
            prompt_config=metadata.prompt_config,
            llm_config=LLMConfig(**metadata.llm_config),
            rag_config=RAGConfig(**metadata.rag_config),
            batch_size=metadata.batch_size,
            clear=False,
            _expect_existing_metadata=True
        )

    def _run_trial(
        self,
        trial_num: int,
        collect_reasoning: bool=False,
        max_threads: int=5,
        api_key: str | None=None
    ) -> Trial:
        """
        Generate scores for all features, for one trial.
        """

        print(f"[INFO] Running trial {trial_num}")

        # division with rounding up:
        num_batches = (len(self.feature_names) + self.metadata.batch_size - 1) // self.metadata.batch_size
        batch_size = (len(self.feature_names) + num_batches - 1) // num_batches

        batches = [
            self.feature_names[i:i+batch_size] for i in range(0, len(self.feature_names), batch_size)
        ]

        # Generate scores in a multithreaded fashion
        score_inputs = [
            GenerateScoresArguments(
                feature_subset=batch,
                prompt_config=PriorGenerationPrompt(**self.metadata.prompt_config),
                rag_config=self.rag_config,
                collect_reasoning=collect_reasoning,
                llm_config=LLMConfig(**self.metadata.llm_config),
                api_key=api_key
            ) for batch in batches
        ]

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            outputs = list(
                executor.map(
                    _generate_scores, score_inputs
                )
            )
            print()

        # From the LLM outputs for each batch, extract scores and reasoning
        # for all features. The structured call in _generate_scores should
        # ensure that all scores are collected. This block also tracks cost.
        trial_result = Trial()
        for output in outputs:
            if output.success:
                llm_out = output.output
                # add this batch to the trial result
                trial_result += Trial(
                    scores=llm_out["scores"],
                    reasoning=llm_out["reasoning"] if collect_reasoning else {},
                    success=True,
                    total_cost=output.cost,
                    raw_outputs=[output]
                )
            else:
                trial_result += Trial(
                    success=False,
                    total_cost=output.cost,
                    raw_outputs=[output]
                )
            self.cost += output.cost
        
        # Save outputs to file
        if trial_result.success:
            output_filename = Path(self.metadata.trial_dir) / f"trial_{trial_num:02d}.json"
            self.metadata.add_trial(output_filename)
        else:
            id = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = Path(self.metadata.failed_trial_dir) / f"trial_at_{id}.json"
            
        trial_result.save(output_filename)
        self.metadata.save()
        self.cost.save(self.cost_filename)

        return trial_result
    
    def generate(
        self,
        collect_reasoning: bool=False,
        num_trials: int=5,
        max_threads: int=5,
        api_key: str | None=None
    ):
        """
        [MAIN GENERATION METHOD] Generate scores for multiple trials and average.
        """
        for i in range(self.metadata.completed_trials, num_trials):
            trial = self._run_trial(
                trial_num=i,
                collect_reasoning=collect_reasoning,
                max_threads=max_threads,
                api_key=api_key,
            )
            if not trial.success:
                print(f"[WARNING] Error in trial {i}. Saving current trials so far and then stopping.")
                break
        
        # Produce final score output and save
        scores = [Trial(
                **json.loads(Path(f).read_text())
            ).scores for f in self.metadata.trial_filenames
        ]
        df = pd.DataFrame(scores)
        df.to_csv(self.metadata.scores_csv, index=False)
        np.save(
            self.metadata.avg_scores_filename,
            df.mean(axis=0).to_numpy()
        )
        print(f"[INFO] Done collecting scores. Results saved to {self.output_dir}.")

    def get_scores(self) -> np.ndarray:
        """
        Returns the average scores generated across trials
        """
        assert Path(self.metadata.avg_scores_filename).exists(), "[FATAL] Scores not yet generated."
        return np.load(self.metadata.avg_scores_filename)


def load_scores(
    generated_prior_dir: str | Path
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Loads trial scores and average per-feature scores for a generated
    prior experiment
    """
    generated_prior_dir = Path(generated_prior_dir)
    metadata_filename = generated_prior_dir / "metadata.json"
    assert metadata_filename.exists(), f"Could not find metadata.json in {generated_prior_dir}"

    metadata = GeneratedPriorMetadata(
        **json.loads(metadata_filename.read_text())
    )
    avg_scores = np.load(metadata.avg_scores_filename)
    scores = pd.read_csv(metadata.scores_csv)

    return avg_scores, scores


def _get_output_format(feature_subset: list[str], collect_reasoning: bool) -> tuple[
    BaseModel, Callable
]:
    """
    Returns a Pydantic BaseModel output format for a batch of features.
    This is used for validating LLM outputs.
    """

    # for now, just make everything lowercase. In the future, we might want
    # to remove all special characters, but that would involve modifying
    # some of the "bag of words" datasets to not have features that are
    # simply "-" or "--" or "_"
    standardized_features = {}
    for key in feature_subset:
        standardized_features[key.lower()] = key

    def validate_dict_keys(d: dict[str, Any]) -> bool:
        key_mapping = {}
        for key in d:
            key_mapping[key.lower()] = key
        if set(key_mapping.keys()) != set(standardized_features.keys()):
            # print which are extra or missing
            extra_keys = set(key_mapping.keys()) - set(standardized_features.keys())
            missing_keys = set(standardized_features.keys()) - set(key_mapping.keys())
            raise ValueError(
                f"LLM output has incorrect keys. "
                f"Extra keys: {extra_keys}, "
                f"Missing keys: {missing_keys}"
            )
        for key in key_mapping.keys():
            if key_mapping[key] != standardized_features[key]:
                d[standardized_features[key]] = d[key_mapping[key]]
                del d[key_mapping[key]]

    def validator(parsed: BaseModel) -> bool:
        validate_dict_keys(parsed.scores)
        if collect_reasoning:
            validate_dict_keys(parsed.reasoning)
        
    output_format = OutputModelReasoning if collect_reasoning else OutputModel
    return output_format, validator


@dataclass
class GenerateScoresArguments:
    """
    Arguments for generating scores for a batch of features.
    """
    feature_subset: list[str]
    rag_config: RAGConfig
    prompt_config: PriorGenerationPrompt
    collect_reasoning: bool
    llm_config: LLMConfig
    api_key: str


def _generate_scores(args: GenerateScoresArguments) -> LLMOutput:
    """
    Generates scores for one batch of features, including:
    1. Building the prompt
    2. Applying RAG context, if applicable
    3. Calling an LLM (with JSON structured outputs)
    """
    feature_subset = args.feature_subset
    
    # Build prompt
    output_instr = OUTPUT_FORMAT_REASONING \
            if args.collect_reasoning else OUTPUT_FORMAT_NO_REASONING
    query = args.prompt_config.fill_in_prompt(
        output_format_instr=output_instr,
        features=feature_subset,
    )

    # Build RAG context
    rag_system = args.rag_config.instantiate_rag_system()
    if rag_system is not None:
        rag_docs = rag_system.retrieve_docs(
            batch_features=args.feature_subset,
            task_description=args.prompt_config.get_task_description()
        )
        rag_context = "\n".join([x.format() for x in rag_docs])
    else:
        rag_context = ""
    
    # Append RAG context to the prompt
    if rag_context != "":
        prompt = fill_in_prompt(
            PROMPT_WITH_RAG, dict(
                query=query,
                context=rag_context
            )
        )
    else:
        prompt = fill_in_prompt(
            PROMPT_WITHOUT_RAG,
            dict(query=query)
        )

    # System prompt and output format
    system_prompt = args.prompt_config.get_system_prompt()
    output_format, validator_fn = _get_output_format(
        feature_subset=feature_subset,
        collect_reasoning=args.collect_reasoning
    )

    # Call the LLM
    llm = LLM(
        config=args.llm_config,
        api_key=args.api_key,
    )
    out = llm.call_structured_json(
        prompt=prompt,
        response_model=output_format,
        system_prompt=system_prompt,
        output_validator=validator_fn,
    )

    # Set output, changing feature names starting with an underscore back to
    # their original values for post-processing
    if out.output is not None:
        out.output = out.output.model_dump()
    print("▒", end="", flush=True) # some visual feedback for progress
    return out