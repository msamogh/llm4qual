# llm4qual

Welcome to the llm4qual repository, your go-to toolkit for enhancing qualitative research with Large Language Models (LLMs). Our mission is to harness the power of LLMs to streamline and support various tasks in qualitative research.

## Supported Pipelines

As of Jan 2024, llm4qual supports a single task (llm-proxy: simulating human annotators in text labeling tasks using LLMs). But we plan to add support for more task types.

### llm-proxy

The `llm-proxy` task is designed to act as a stand-in for human annotators in text-annotation tasks. By leveraging LLMs, `llm-proxy` provides a sophisticated and automated approach to evaluating qualitative data across multiple dimensions.

## Features

- **Dataset Loading**: Integrate seamlessly with Langchain + HuggingFace to import your qualitative datasets.
- **Prompt Templating**: Organize your annotation prompts with ease using our structured YAML file templates, compatible with Langchain.
- **Few-Shot Examples**: Incorporate few-shot learning by providing examples within your YAML files to enhance the LLM's annotation accuracy.
- **Evaluation Suite Setup**: Utilize helper classes and functions to establish a HuggingFace `EvaluationSuite` tailored to your research needs.
- **Result Reporting**: After analysis, neatly save and report the evaluation outcomes for further review and action.

## Getting Started

To get started with llm4qual, follow these simple steps:

### Prerequisites

Ensure you have the latest versions of Langchain and HuggingFace libraries installed. You can install these using pip:

```bash
pip install langchain huggingface_hub
```

## Installation
1. Clone the LLM4Qual repository to your local machine:
```bash
git clone https://github.com/msamogh/llm4qual.git
cd llm4qual
```

2. Navigate to the llm-proxy directory and install any necessary dependencies:
```bash
cd llm-proxy
poetry shell
poetry install
```

3. Install Transformers4Qual:
```bash
git clone https://github.com/msamogh/transformers4qual.git
pip install -e transformers4qual
```

4. Install Evaluate4Qual:
```bash
git clone https://github.com/msamogh/evaluate4qual.git
pip install -e evaluate4qual
```

## Usage
1. Load Your Dataset: Use Langchain to load your dataset into the framework.
2. Create Prompt Templates: Define your annotation dimensions and write prompt templates in YAML format.
3. Few-Shot Examples: Add few-shot examples to your YAML files to guide the LLM in annotation tasks.
4. Initialize Evaluation Suite: With the provided helper classes, set up your EvaluationSuite.
5. Run Evaluation: Execute the evaluation and collect results.
6. Save & Report: Utilize our reporting tools to save and visualize the analysis outcomes.

## License
Distributed under the MIT License. See LICENSE for more information.

## Contact
For any additional questions or comments, please reach out to msamogh@gmail.com or submit an issue.
