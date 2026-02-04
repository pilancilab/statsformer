#!/usr/bin/env python3

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from transformers.hf_argparser import HfArgumentParser

from statsformer.data.dataset import Dataset
from statsformer.llm.generated_prior import GeneratedPrior
from statsformer.llm.common import LLMConfig
from statsformer.llm.prompting import PriorGenerationPrompt
from statsformer.llm.rag.base import RAGConfig
from dotenv import load_dotenv



@dataclass
class Arguments:
    dataset_dir: str = field(metadata=dict(
        help="Path to the dataset directory."
    ))
    data_dir: str = field(default="data", metadata=dict(
        help="Base directory for dataset input/output (default: 'data')."
    ))
    experiment_name: str = field(
        default="scores", metadata=dict(
        help=(
            "Name for the experiment, used in creating the output directory. Need not include any "
            "information about the model/temperature/prompts/task/RAG/etc, as the output directory "
            "naming scheme already includes such info."
        ))
    )
    batch_size: int = field(default=30, metadata=dict(
        help="Number of features to pass into each API call."
    ))
    num_trials: int = field(default=5, metadata=dict(
        help="Number of generation trials per feature.",
    ))
    collect_reasoning: bool = field(default=False, metadata=dict(
        help="Whether to collect reasoning from the model.",
    ))
    max_threads: int = field(default=5, metadata=dict(
        help="Maximum number of threads for parallel generation.",
    ))
    clear: bool = field(default=False, metadata=dict(
        help="Clear existing outputs before generation.",
    ))
    


def main(
    args: Arguments,
    prompt_config: PriorGenerationPrompt,
    llm_config: LLMConfig,
    rag_config: RAGConfig,
    output_dir: str
):
    load_dotenv()

    dataset = Dataset.from_dir(args.dataset_dir)
    feature_names = dataset.feature_names()
    generated_priors = GeneratedPrior(
        feature_names=feature_names,
        output_dir=output_dir,
        prompt_config=prompt_config,
        llm_config=llm_config,
        rag_config=rag_config,
        batch_size=args.batch_size,
        clear=args.clear,
    )

    generated_priors.generate(
        collect_reasoning=args.collect_reasoning,
        num_trials=args.num_trials,
        max_threads=args.max_threads,
        api_key=llm_config.get_key_from_env(),
    )


def parse_args() -> tuple[
    Arguments, PriorGenerationPrompt, LLMConfig, RAGConfig
]:
    parser = HfArgumentParser([Arguments, PriorGenerationPrompt, LLMConfig, RAGConfig])
    return parser.parse_args_into_dataclasses()


if __name__ == "__main__":
    (args, prompt_config, llm_config, rag_config) = parse_args()

    dataset_name = Path(args.dataset_dir).name
    prompt_name = Path(prompt_config.prompt_filename).stem

    output_dir = Path(args.data_dir) / "generated_priors" / \
        Path(args.dataset_dir).name / \
        f"RAG_{rag_config.rag_type}" / \
        f"{llm_config.model_name.replace('/', '_')}__temp_{llm_config.temperature}__batch_{args.batch_size}" / \
        f"{Path(prompt_config.prompt_filename).stem}__{Path(prompt_config.system_prompt_filename).stem}" / \
        args.experiment_name

    main(
        args=args,
        prompt_config=prompt_config,
        llm_config=llm_config,
        rag_config=rag_config,
        output_dir=output_dir
    )
