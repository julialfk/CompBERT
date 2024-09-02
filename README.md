# CompBERT

This repository contains the implementation of a language model, named CompBERT, designed to facilitate the assessment of functional completeness in software projects. CompBERT helps users identify which methods or functions within a codebase are likely related to a given natural language (NL) feature description. This project builds upon Microsoft's [UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) model and code search approach for fine-tuning.

### Key Features

- **Custom dataset creation**: The project includes a method for creating datasets from IlmSeven and SEOSS open-source projects.
- **Fine-tuning datasets**: The full training, development and evaluation datasets constructed and used for this project can be found in a separate [HuggingFace repository](https://huggingface.co/datasets/JuliaLFK/CompBERT_data).
- **Fine-tuning**: The scripts and configurations used for fine-tuning the UniXcoder model to create CompBERT.
- **Output models**: The models created used in the experiments of this thesis can be found in a separate [HuggingFace repository](https://huggingface.co/JuliaLFK/CompBERT/). These models are fine-tuned to score each method, indicating its relevance to a given NL feature description.
