from openai import OpenAI
from statsformer.llm.common import LLMConfig, LLMProvider
from statsformer.llm.pricing.openai import OPENAI_STANDARD_PRICING
from statsformer.utils import dataclass_to_json


from dataclasses import dataclass, field


INPUT_COST_KEY = "input_cost_per_1M"
OUTPUT_COST_KEY = "output_cost_per_1M"


class ModelCosts:
    def __init__(
        self,
        llm_config: LLMConfig,
        api_key: str | None=None,
    ):
        if llm_config.llm_provider == LLMProvider.OPENROUTER.value:
            self.cost = _get_cost_openrouter(
                llm_config.get_client(api_key),
                model_name=llm_config.model_name)
        elif llm_config.llm_provider == LLMProvider.OPENAI.value:
            self.cost = OPENAI_STANDARD_PRICING.get(llm_config.model_name, {})
        
        if not self.cost:
            print(f"[WARNING] Could not determine model cost for {llm_config.llm_provider} model {llm_config.model_name}, cost tracking disabled.")
        
    def get_cost(
        self, input_tokens: int, output_tokens: int
    ):
        """
        INTERNAL: Get the (input, output) cost of a request based on input and
        output tokens.
        """
        if not self.cost:
            return 0.0, 0.0
        return (
            self.cost.get(INPUT_COST_KEY, 0.0) * input_tokens / 1e6,
            self.cost.get(OUTPUT_COST_KEY, 0.0) * output_tokens / 1e6
        )


@dataclass
class LLMCost:
    input_tokens: int = field(default=0)
    output_tokens: int = field(default=0)
    input_cost: float = field(default=0)
    output_cost: float = field(default=0)

    def __add__(self, other: "LLMCost"):
        return LLMCost(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost
        )

    def total_cost(self):
        return self.input_cost + self.output_cost

    def save(self, filename: str):
        return dataclass_to_json(self, filename)


def _get_cost_openrouter(
    client: OpenAI,
    model_name: str,
) -> float:
    models = client.models.list()
    for model in models:
        if model.id == model_name:
            try:
                pricing = model.pricing
                return {
                    INPUT_COST_KEY: float(pricing["prompt"]) * 1e6,
                    OUTPUT_COST_KEY: float(pricing["completion"]) * 1e6
                }
            except Exception as e:
                return {}
    return {}