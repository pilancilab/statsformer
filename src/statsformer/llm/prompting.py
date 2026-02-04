from dataclasses import dataclass, field
import json
from pathlib import Path
import re


@dataclass
class TaskDescription:
    context: str
    task: str
    notes: str | None = field(default=None)


@dataclass
class PriorGenerationPrompt:
    """
    Handles loading and filling in prompt templates for prior generation tasks.
    """
    prompt_filename: str = field(metadata=dict(
        help="Prompt template filename."
    ))
    system_prompt_filename: str = field(metadata=dict(
        help="System prompt template filename."
    ))
    task_filename: str = field(metadata=dict(
        help="Path to a JSON with \"context\" and \"task\" fields describing the prediction task."
    ))

    def get_task_description(self):
        return TaskDescription(**json.loads(
            Path(self.task_filename).read_text()
        ))

    def fill_in_prompt(
        self, output_format_instr: str,
        features: list[str],
    ) -> str:
        """
        Produces a filled-in prompt for the prior generation task.
        """
        task_desc = self.get_task_description()
        return fill_in_prompt_file(
            template_path=self.prompt_filename,
            arguments=dict(
                context=task_desc.context,
                task=task_desc.task,
                output_format=output_format_instr,
                features=str(features),
                optional_notes=f"**Notes**:\n{task_desc.notes}" \
                    if task_desc.notes is not None else ""
            )
        )
    
    def get_system_prompt(self):
        return Path(self.system_prompt_filename).read_text()


def fill_in_prompt(
    template: str,
    arguments: dict[str, str]
):
    """
    Fills in a prompt template with the given arguments.
    Placeholders in the template should be in the format {{key}} where key is
    a key in the arguments dictionary.
    """
    for key, value in arguments.items():
        template = template.replace("{{" + key + "}}", value)

    # Check for any unreplaced placeholders
    unreplaced = re.findall(r"{{(.*?)}}", template)
    if unreplaced:
        print(f"[WARNING] Unreplaced placeholders in template: {unreplaced}")
    return template


def fill_in_prompt_file(
    template_path: str,
    arguments: dict[str, str]
):
    template = Path(template_path).read_text()
    return fill_in_prompt(template, arguments)


###############################################################################
############################# PROMPTING CONSTANTS #############################
###############################################################################

# Output format instructions for LLM responses
OUTPUT_FORMAT_NO_REASONING = """
{
    "scores": {
        "FEATURE_NAME_01": floating_point_score_value,
        "FEATURE_NAME_02": floating_point_score_value,
        ...one score per feature name.
    }
}
"""

OUTPUT_FORMAT_REASONING = """
{
    "scores": {
        "FEATURE_NAME_01": floating_point_score_value,
        "FEATURE_NAME_02": floating_point_score_value,
        ...one score per feature name.
    },
    "reasoning":  "scores": {
        "FEATURE_NAME_01": "summary of your reasoning for this feature",
        "FEATURE_NAME_02": "summary of your reasoning for this feature",
        ...one reasoning field per feature name.
    }
}
"""

# Prompts for LLM feature scoring with and without RAG
PROMPT_WITH_RAG = """Using the following context, provide the most accurate and relevant answer to the question.
Prioritize the provided context, but if the context does not contain enough information to fully address the question,
use your best general knowledge to complete the answer:

{{context}}

====================
Question: 
====================
{{query}}
"""

PROMPT_WITHOUT_RAG="""Using your best general knowledge, provide the most accurate and relevant answer to the question:
====================
Question: 
====================
{{query}}
"""
