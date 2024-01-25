from typing import *
from dataclasses import dataclass
from functools import partial
import json
import os

from langchain.prompts import load_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator

from transformers.pipelines import LangchainModelForProxyLLM, LangchainConfig

import evaluate
import datasets


RAG = datasets.splits.NamedSplit("rag")


def get_prompt_template(
    prompts_root_dir: Text,
    suite_name: Text,
    rubric_name: Text,
    prompt_template_name: Text,
) -> Text:
    rubric_directory = "/".join(rubric_name.split("."))
    return load_prompt(
        f"{prompts_root_dir}/{suite_name}/rubrics/{rubric_directory}/{prompt_template_name}.yaml"
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
        split_str: Text,
        rubrics_to_prompt_templates: Mapping[Text, Sequence[Text]],
        process_predictions_fn_or_map: Union[
            Mapping[Text, Callable], Callable
        ] = lambda x: x,
    ):
        super().__init__(suite_name)

        self.suite_name = suite_name
        self.rubrics_to_prompt_templates = rubrics_to_prompt_templates
        self.rubrics_to_input_variables = extract_input_variables_from_prompt_template(
            rubrics_to_prompt_templates,
            prompts_root_dir="prompts",
            suite_name=suite_name,
        )
        self.suite = self._create_subtasks(
            data, split_str, metric, process_predictions_fn_or_map, suite_name
        )

    def evaluate_rubric_with_single_prompt(
        self,
        prompts_dir: Text,
        rubric_name: Text,
        prompt_name: Text,
        model_name: Text = "gpt-3.5-turbo",
        temperature: float = 0.1,
        return_dict: bool = False,
        mock_llm_call: bool = False,
    ) -> Dict:
        runnable = self._prepare_runnable(
            prompts_dir, rubric_name, prompt_name, model_name, temperature
        )
        pipeline = LangchainModelForProxyLLM(
            LangchainConfig(runnable=runnable, mock_llm_call=mock_llm_call)
        )
        return self._run_and_format_results(rubric_name, pipeline, return_dict)

    def _run_and_format_results(self, rubric_name, pipeline, return_dict):
        results = super().run_task_wise(rubric_name, pipeline, return_predictions=True)
        if return_dict:
            return {f"{rubric_name}": results}
        else:
            return results

    @staticmethod
    def _prepare_runnable(
        prompts_dir: Text,
        rubric_name: Text,
        prompt_name: Text,
        model_name: Text,
        temperature: float,
    ) -> Any:
        runnable = (
            load_prompt(
                f"prompts/{prompts_dir}/rubrics/{rubric_name.replace('.', '/')}/{prompt_name}.yaml"
            )
            | ChatOpenAI(model_name=model_name, temperature=temperature)
            | StrOutputParser()
        )
        return runnable

    def _create_subtasks(
        self, data, split_str, metric, process_predictions_fn_or_map, suite_name
    ):
        return [
            evaluate.evaluation_suite.SubTask(
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
                    "predictions_processor_fn": process_predictions_fn_or_map[rubric]
                    if isinstance(process_predictions_fn_or_map, Mapping)
                    else process_predictions_fn,
                },
            )
            for rubric, input_variables in self.rubrics_to_input_variables.items()
            for prompt_template_name in self.rubrics_to_prompt_templates[rubric]
        ]

    @staticmethod
    def evaluate_all_and_write_results(
        suite_name: Text,
        rubrics_to_prompt_templates: Mapping[Text, Any],
        id_column: Text,
        metric_or_metric_map: Union[Mapping[Text, Text], Text],
        split_str: Text,
        data: Union[Text, datasets.Dataset],
        prompts_dir: Text,
        model_name: Text,
        process_predictions_fn_or_map: Union[Mapping[Text, Callable], Callable] = lambda x: x,
        **evaluator_kwargs,
    ):
        for rubric_name in rubrics_to_prompt_templates.keys():
            for prompt_name in rubrics_to_prompt_templates[rubric_name]:
                print(f"Evaluating {rubric_name}/{prompt_name}")

                metric = (
                    metric_or_metric_map[rubric_name]
                    if isinstance(metric_or_metric_map, Mapping)
                    else metric_or_metric_map
                )

                evaluation_suite = LLMProxyEvaluationSuite(
                    suite_name=suite_name,
                    metric=metric,
                    data=data,
                    split_str=split_str,
                    rubrics_to_prompt_templates=rubrics_to_prompt_templates,
                    process_predictions_fn_or_map=process_predictions_fn_or_map,
                )
                print(f"Initialized {evaluation_suite.suite_name}")

                results = evaluation_suite.evaluate_rubric_with_single_prompt(
                    prompts_dir=prompts_dir,
                    rubric_name=rubric_name,
                    prompt_name=prompt_name,
                    model_name=model_name,
                    **evaluator_kwargs,
                )

                output = {
                    "results": [
                        {"id": data_id, "prediction": prediction, "output": output_data}
                        for data_id, prediction, output_data in zip(
                            results.pop("data")[id_column],
                            results.pop("predictions"),
                            results.pop("outputs"),
                        )
                    ],
                    **results,
                }
                output["raw_predictions"] = [x["prediction"] for x in output["results"]]

                # Ensure the results directory exists
                results_dir = os.path.join("results", evaluation_suite.suite_name)
                os.makedirs(results_dir, exist_ok=True)

                # Write the results to a JSON file
                results_file_path = os.path.join(
                    results_dir, f"{rubric_name}_{prompt_name}_{model_name}.json"
                )
                with open(results_file_path, "w") as file:
                    json.dump(output, file, indent=2)

                print(f"Results written to {results_file_path}")
