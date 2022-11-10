# Amber Machine Learning Engineer Exercise (Oct 2022)

A multistep power comnsuption prediction model for a single customer. Takes in the last 24 hours of power consumption data and predicts the next 24 hours of power consumption, in 5min intervals

## Quick Start

----

1. Setup Environment: `make setup-dev`
    - if you prefer to use a docker container instead of installing stuff locally, run `make docker-dev-shell`. This will load a docker container environemnt with everything setup for you.
2. Run the ML pipeline End to End: `make run-pipeline`
   - I've created a very basic pipeline using a tool called dvc, that executes ML workflow steps as a DAG, which looks like this (`make show-pipeline`):

    ```bash
    •••• make show-pipeline
    dvc dag
                +--------------+
                | data/raw.dvc |
                +--------------+
                        *
                        *
                        *
                +------------+
                | clean-data |
                +------------+
                        *
                        *
                        *
                +------------------+
                | generate-dataset |
                +------------------+
                ***            ***
                **                  **
            **                      **
    +-------------+                     **
    | train-model |                   **
    +-------------+                 **
                ***            ***
                    **        **
                    **    **
                +--------------+
                | predict-powr |
                +--------------+
    ```

    - It tracks & versions artifacts too!
    - Pipeline execution follows the following `dvc -> dvc <stage> -> make <target> -> python main.py <subcommand>` flow:
      - `dvc` is used to track & version artifacts
      - `make` used to automate common development tasks, like running the pipeline, showing the pipeline, running tests, running a particular ML pipeline step, etc.
      - `python main.py` is used to execute the ML pipeline steps
3. Run `make help` to see all the available make targets
4. Run `python3 main.py --help` to see all the available subcommands
5. I've jotted down my thoughts during initial exploration of the data & modelling within their respective notebooks `notebooks/*`. It's a bit messy, but it's a good place to start if you're interested in my thought process. And docstrings within the source code should summarize the process too. I am happy to walk through my thought process & and this source code during the next stages!

### Directory Structure

```bash
.
├── .github                <- github Actions
├── config                  <-- config files
├── data                   <-- root data directory
│   ├── clean               <-- contains clean data
│   ├── dataset             <-- generated datasets
│   ├── predictions         <-- predictions
│   └── raw                 <-- raw data
├── docker                 <-- docker related files
├── models                 <-- trained models
│   └── linear_model        <-- linear model for power consumption prediction
│       ├── assets
│       └── variables
├── notebooks             <-- jupyter notebooks used in development
│   └── archive
├── powr                  <-- python package that contains source code
├── tests                   <-- test files
│   └── code
```
