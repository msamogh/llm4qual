from typing import *
from dataclasses import dataclass, field

from datasets.features import ClassLabel, Features, Value
import datasets.features as features
from datasets.tasks.base import TaskTemplate

import evaluate
import datasets
from evaluate.evaluation_suite import SubTask


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
            LLMProxyEvaluationSuite.extract_input_variables_from_prompt_template(
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
                    if self.task_type == "llm-proxy-regression"
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
