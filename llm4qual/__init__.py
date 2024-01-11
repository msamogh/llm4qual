from typing import *
from dataclasses import dataclass, field

from transformers import Pipeline
from datasets.features import ClassLabel, Features, Value
from datasets.tasks.base import TaskTemplate
from langchain.prompts import load_prompt
from evaluate.evaluation_suite import SubTask
import evaluate
import datasets
import datasets.features as features


def extract_input_variables_from_prompt_template(
    rubrics_to_prompt_templates: Mapping[Text, Sequence[Text]],
    prompts_root_dir: Text,
    suite_name: Text,
) -> Sequence[Text]:
    """Find the union of all input features over all prompt templates for each rubric."""
    rubrics_to_input_variables = {}
    for rubric, prompt_template_names in rubrics_to_prompt_templates.items():
        input_variables = set()
        for prompt_template_name in prompt_template_names:
            for input_variable in load_prompt(
                f"{prompts_root_dir}/{suite_name}/rubrics/{rubric.split('.')[0]}/{rubric.split('.')[1]}/{prompt_template_name}.yaml",
            ).input_variables:
                input_variables.add(input_variable)
        rubrics_to_input_variables[rubric] = list(input_variables)
    return rubrics_to_input_variables


class LLMProxyConfig(datasets.BuilderConfig):
    def __init__(self, name, version, description, features, **kwargs):
        super().__init__(name=name, version=version, description=description, **kwargs)
        self.features = features


@dataclass
class LLMProxyEvaluationSuite(evaluate.EvaluationSuite):
    def __init__(
        self,
        *,
        suite_name: Text,
        task_type: Text,
        dataset: datasets.Dataset,
        split_str: Text = "val_test[:40]",
        rubrics_to_prompt_templates: Mapping[Text, Sequence[Text]],
    ):
        super().__init__(suite_name)

        self.suite_name = suite_name
        self.rubrics_to_prompt_templates = rubrics_to_prompt_templates
        self.rubrics_to_input_variables = (
            extract_input_variables_from_prompt_template(
                rubrics_to_prompt_templates,
                prompts_root_dir="prompts",
                suite_name=suite_name,
            )
        )
        self.evaluation_suite = [
            SubTask(
                task_type=task_type,
                data=dataset,
                subset=rubric,
                split=split_str,
                args_for_task={
                    "metric": "mse"
                    if task_type == "llm-proxy-regression"
                    else "accuracy",
                    "input_variables_column": input_variables,
                    "label_column": "label",
                },
            )
            for rubric, input_variables in self.rubrics_to_input_variables.items()
        ]


class LLMProxyRegressionTask(TaskTemplate):

    task: str = field(
        default="llm-proxy-regression",
        metadata={"include_in_asdict_even_if_is_default": True},
    )
    input_schema: ClassVar[Features] = Features(
        {"input_variables": features.Sequence(feature={"text": "string"})}
    )
    label_schema: ClassVar[Features] = Features({"label_values": ClassLabel})
    input_variables_column: str = "input_variables"
    label_column: str = "label_values"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {
            self.input_variables_column: "input_variables",
            self.label_column: "label_values",
        }


class LLMProxyClassificationTask(TaskTemplate):

    task: str = field(
        default="llm-proxy-classification",
        metadata={"include_in_asdict_even_if_is_default": True},
    )
    input_schema: ClassVar[Features] = Features(
        {"input_variables": features.Sequence(feature={"text": "string"})}
    )
    label_schema: ClassVar[Features] = Features({"labels": Value})
    input_variables_column: str = "input_variables"
    label_column: str = "labels"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {
            self.input_variables_column: "input_variables",
            self.label_column: "labels",
        }


class LangchainPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        print(f"SANITIZING {kwargs}")
        if "chain" not in kwargs:
            raise ValueError("chain must be specified")
        self.chain = kwargs.pop("chain")

    def preprocess(self, inputs: Dict[Text, Any]) -> Dict[Text, Any]:
        print(f"PREPROCESSING {inputs}")
        return inputs

    def postprocess(self, model_outputs):
        print(f"POSTPROCESSING {model_outputs}")
        return model_outputs

    def _forward(self, model_inputs):
        print(f"FORWARDING {model_inputs}")
        output = self.chain.invoke(model_inputs)
        return output
