from typing import *
from dataclasses import dataclass, field
from functools import partial

from datasets.features import ClassLabel, Features, Value
from datasets.tasks.base import TaskTemplate
from langchain.prompts import load_prompt
from langchain_openai import ChatOpenAI
import evaluate
import datasets
import datasets.features as features


def get_prompt_template(
    prompts_root_dir: Text,
    suite_name: Text,
    rubric_name: Text,
    prompt_template_name: Text,
) -> Text:
    return load_prompt(
        f"{prompts_root_dir}/{suite_name}/rubrics/{rubric_name.split('.')[0]}/{rubric_name.split('.')[1]}/{prompt_template_name}.yaml"
    )


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
            for input_variable in get_prompt_template(
                prompts_root_dir=prompts_root_dir,
                suite_name=suite_name,
                rubric_name=rubric,
                prompt_template_name=prompt_template_name,
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
    prompts_root_dir: Text,
    suite_name: Text,
    rubric_name: Text,
    prompt_template_name: Text,
) -> Text:
    """Populate the langchain template with the example's input variables."""
    prompt_template = get_prompt_template(
        prompts_root_dir=prompts_root_dir,
        suite_name=suite_name,
        rubric_name=rubric_name,
        prompt_template_name=prompt_template_name,
    )
    return {
        **{k: sample[k] for k in prompt_template.input_variables},
        "label_column": rubric_name,
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
        from evaluate.evaluation_suite import SubTask

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
                task_type="llm-proxy",
                data=data,
                subset=rubric,
                split=split_str,
                data_preprocessor=partial(
                    get_prompt,
                    prompts_root_dir="prompts",
                    suite_name=suite_name,
                    rubric_name=rubric,
                    prompt_template_name=prompt_template_name,
                ),
                args_for_task={
                    "metric": metric,
                    "input_variables": get_prompt_template(
                        prompts_root_dir="prompts",
                        suite_name=suite_name,
                        rubric_name=rubric,
                        prompt_template_name=prompt_template_name,
                    ).input_variables,
                    "label_column": rubric,
                },
            )
            for rubric, input_variables in self.rubrics_to_input_variables.items()
            for prompt_template_name in rubrics_to_prompt_templates[rubric]
        ]

    def evaluate_rubric_with_single_prompt(
        self,
        prompts_dir: Text,
        rubric_name: Text,
        prompt_name: Text,
        model_name: Text = "gpt-3.5-turbo",
        temperature: float = 0.1,
        return_dict: bool = False,
    ) -> Dict:
        from transformers.pipelines import LangchainModelForProxyLLM, LangchainConfig

        runnable = load_prompt(
            f"prompts/{prompts_dir}/rubrics/{rubric_name.replace('.', '/')}/{prompt_name}.yaml"
        ) | ChatOpenAI(model_name=model_name, temperature=temperature)
        pipeline = LangchainModelForProxyLLM(LangchainConfig(runnable=runnable))
        return (
            {f"{rubric_name}/{prompt_name}": super().run(pipeline)}
            if return_dict
            else super().run(pipeline)
        )

    def evaluate_rubric_with_all_prompts(
        self,
        prompts_dir: Text,
        rubric_name: Text,
        model_name: Text = "gpt-3.5-turbo",
        temperature: float = 0.1,
    ) -> Dict:
        return {
            f"{rubric_name}/{prompt_name}": self.run_single_prompt(
                prompts_dir=prompts_dir,
                rubric_name=rubric_name,
                prompt_name=prompt_name,
                model_name=model_name,
                temperature=temperature,
            )
            for prompt_name in self.rubrics_to_prompt_templates[rubric_name]
        }

    def evaluate_all_rubrics_with_all_prompts(
        self,
        prompts_dir: Text,
        model_name: Text = "gpt-3.5-turbo",
        temperature: float = 0.1,
    ) -> Dict:
        return {
            f"{rubric_name}/{prompt_name}": self.evaluate_rubric_with_all_prompts(
                prompts_dir=prompts_dir,
                rubric_name=rubric_name,
                model_name=model_name,
                temperature=temperature,
            )
            for rubric_name in self.rubrics_to_prompt_templates.keys()
            for prompt_name in self.rubrics_to_prompt_templates[rubric_name]
        }
