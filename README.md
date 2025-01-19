# Analyzing-Promises-and-Constraint-Claims

## Overview
This project explores trends in sustainability reporting from 2020 to 2024 using a dataset of 6,500 reports across four key sectors: natural resources, manufacturing, retail, and information. The study employs a fine-tuned ClimateBERT model to classify sustainability claims as "Promises," "Constraints," or "Neutral." Key outputs include annotated datasets, fine-tuned ClimateBERT-Promise-Constraint model, and topic analysis of constrains.

This repository contains the code and data for the project. The code is organized into the following directories:
- `data/`: Contains the annotated dataset and the topics for the constraints claims.
- `src/`: Contains four Python scripts for model training, inference, RAG pipeline for distillation and topic analysis.

This repository provides the overview scripts used for the project. The data pre- and post-processing scripts are available upon request.
If you intend to use the ```topics_assignment.py``` script, please extract the embeddings from the climatebert_promise_constraint model and save them as ```embeddings.pkl``` file for the future use. We do not provide the embeddings file due to its large size.

Further models and data can be found in the following repositories:
- [ClimateBERT-Promise-Constraint](https://huggingface.co/dariast/climatebert_promise_constraint)
- [Climate-Promise-Constraint Dataset ](https://huggingface.co/datasets/dariast/Climate-Promise-Constraint)
---

## Features
- **Dataset**: 5,386 annotated text samples derived from publicly available sustainability reports.
- **Model**: ClimateBERT fine-tuned for promise-constraint classification.
- **Sectoral Analysis**: Insights into trends across four key sectors.
- **Topic Modeling**: Identification of key constraints and their evolution.

---

## Results
Key findings include:
- Distribution of constraint claims across sectors and years.
- High-performance metrics for the fine-tuned ClimateBERT model (e.g., F1 Macro Score: ~88%).
- Topic analysis highlighting key challenges like "Market Risks" and "Operational Obstacles."

---



## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

