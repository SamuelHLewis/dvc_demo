# DVC Demo

This repo demonstrates how to use DVC to manage evaluation pipelines for AI products and components.

## Installation
Install with uv

## Repo Structure

The repo has 5 main files:
1. `app.py`: a barebones app which writes a limerick on a given topic
2. `judge.py`: an LLM-as-judge which scores the responses from `app.py`, implemented with pydantic evals.
3. `topics.txt`: a list of topics for limericks.
4. `params.yaml`: the values for all variables used at each stage of the pipeline.
5. `dvc.yaml`: the specification of the eval pipeline.

## Pipeline Visualisation
To visualise the pipeline, run:
```bash
dvc dag
```
This will produce a mermaid-format directed acyclic graph of the pipeline stages.

## Running DVC
### Version-aware
To run the pipeline in a version-aware manner, run:
```bash
uv run dvc repro
```
This is the primary use case for DVC: it checks the inputs and dependencies of each stage, and only runs the stages that have changed since the last run.

### End-to-end
To run the pipeline end-to-end, regardless of changes to inputs and dependencies, run:
```bash
uv run dvc repro --force
```
This is useful if you always want to run the most recent version of the pipeline e.g. in a regression test