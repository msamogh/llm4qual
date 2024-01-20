from typing import *
from fire import Fire
import yaml
import csv

from datasets import load_dataset


def convert(
    dataset_script: Text,
    id_column_name: Text,
    rag_ids_file_path: Text,
    output_file_path: Text,
    output_file_name: Text = "examples.yaml",
):
    dataset = load_dataset(dataset_script)["val_test"]
    with open(rag_ids_file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            rubric, sample_ids = row
            rag_ids = sample_ids.split(",")
            rag_projects = dataset.filter(lambda x: x[id_column_name] in rag_ids)
            output_dicts = []
            for project in rag_projects:
                output_dicts.append(project)
            
            rubric_path_element = rubric.replace(".", "/")
            path = f"{output_file_path}/{rubric_path_element}/{output_file_name}"
            with open(path, "w") as outfile:
                yaml.dump(output_dicts, outfile)


if __name__ == "__main__":
    convert(
        "amby/dataset.py",
        "project_name",
        "data/amby/rag_ids/rag_ids_1.csv",
        "prompts/amby/examples",
    )
