from typing import *
from dataclasses import dataclass, field

from transformers import Pipeline
from datasets.features import ClassLabel, Features, Value
from datasets.tasks.base import TaskTemplate
from langchain.prompts import load_prompt
from langchain_openai import ChatOpenAI
from evaluate.evaluation_suite import SubTask
import evaluate
import datasets
import datasets.features as features

from .model import LangchainModel


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


def get_prompt(
    sample: Any,
) -> Text:
    """Populate the langchain template with the example's input variables."""
    return {
        "prompt": "sample_text",
        "labels": "sample_label",
    }



@dataclass
class LLMProxyEvaluationSuite(evaluate.EvaluationSuite):
    def __init__(
        self,
        *,
        suite_name: Text,
        metric: Text,
        data: Text,
        split_str: Text = "val_test[:40]",
        rubrics_to_prompt_templates: Mapping[Text, Sequence[Text]],
    ):
        super().__init__(suite_name)

        self.suite_name = suite_name
        self.rubrics_to_prompt_templates = rubrics_to_prompt_templates
        self.rubrics_to_input_variables = extract_input_variables_from_prompt_template(
            rubrics_to_prompt_templates,
            prompts_root_dir="prompts",
            suite_name=suite_name,
        )
        self.suite = [
            SubTask(
                task_type="text-classification",
                data=data,
                subset=rubric,
                split=split_str,
                data_preprocessor=get_prompt,
                args_for_task={
                    "metric": metric,
                    "input_column": "prompt",
                    "label_column": "labels",
                },
            )
            for rubric, input_variables in self.rubrics_to_input_variables.items()
        ]

    def run_single_prompt(
        self,
        prompts_dir: Text,
        rubric_name: Text,
        prompt_name: Text,
        model_name: Text = "gpt-3.5-turbo",
        temperature: float = 0.1,
    ) -> Dict:
        runnable = load_prompt(
            f"prompts/{prompts_dir}/rubrics/{rubric_name.replace('.', '/')}/{prompt_name}.yaml"
        ) | ChatOpenAI(model_name=model_name, temperature=temperature)
        pipeline = LangchainModel(runnable)
        return super().run(pipeline)

    def run_all_prompts(
        self,
        prompts_dir: Text,
        rubric_name: Text,
        model_name: Text = "gpt-3.5-turbo",
        temperature: float = 0.1,
    ) -> Dict:
        return {
            prompt_name: self.run_single_prompt(
                prompts_dir=prompts_dir,
                rubric_name=rubric_name,
                prompt_name=prompt_name,
                model_name=model_name,
                temperature=temperature,
            )
            for prompt_name in self.rubrics_to_prompt_templates[rubric_name]
        }
