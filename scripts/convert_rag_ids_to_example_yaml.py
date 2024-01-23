from typing import *
from collections import defaultdict
import yaml
import csv

from fire import Fire
from datasets import load_dataset


def convert(
    dataset_script: Text,
    id_column_name: Text,
    rag_ids_file_path: Text,
    output_file_path: Text,
    output_file_name: Text = "examples.yaml",
):
    dataset = load_dataset(dataset_script)["rag"]

    rubric_wise_rag_projects = defaultdict(list)
    rubric_wise_output_dicts = defaultdict(list)

    with open(rag_ids_file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip header
        for row in reader:
            rubric, project_name, _, _ = row
            rubric_wise_rag_projects[rubric].append(project_name)

        for rubric, project_names in rubric_wise_rag_projects.items():
            # filter dataset to only include projects in the rubric
            rubric_wise_rag_projects = dataset.filter(
                lambda x: x[id_column_name] in project_names
            )
            for project in rubric_wise_rag_projects:
                rubric_wise_output_dicts[rubric].append(project)

            rubric_path_element = rubric.replace(".", "/")
            path = f"{output_file_path}/{rubric_path_element}/{output_file_name}"
            with open(path, "w") as outfile:
                yaml.dump(rubric_wise_output_dicts[rubric], outfile)


if __name__ == "__main__":
    convert(
        "amby/dataset.py",
        "project_name",
        "data/amby/rag_ids/simple.csv",
        "prompts/amby/examples",
    )
