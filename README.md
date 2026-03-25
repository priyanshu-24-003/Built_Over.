# Built_Over 

## About
Built_Over. is a derivative Repository of forked repo palmer-penguins-classification ,This Repo uses MLOPs concepts OOPs, data versioning, experiment tracking, continuous Integration and few more.

## Purpose

The goal was **not** just building another model — but demonstrating how to implement following **Mlops Concepts** :


| Area                        | Tool/Technique                          | What was implemented                                                                 |
|-----------------------------|-----------------------------------------|--------------------------------------------------------------------------------------|
| 📂 Data & Pipeline Versioning | **DVC**                                | Versioned raw & processed data, automated reproducible pipelines (`dvc repro`)      |
| 📊 Experiment Tracking       | **MLflow**                             | In model_evaluation component added tracking server , logged metric, artifact and models — viewable in MLflow UI       |
| 📦 Remote Data & Model Store  | **DagsHub**                            | remote storage + MLflow remote tracking server                                   |
| 🧪 Continuous Integration    | **GitHub Actions**                     | Automated tests for components, pipeline validation, logger checks                  |
| 🐳 Containerization          | **Docker**                             | Built Docker image for app.py + pushed image to dockerhub                       |
| 🛠 Clean Code Architecture   | **OOP principles**                     | Modular helpers: `Loader`, `Saver`, `Logster` + inheritance usage                   |


## Extra things and important links

- To check remote server visit :https://dagshub.com/priyanshu24003/Built_Over..mlflow
- There is a seperate branch remote_server in this repository
- To pull the docker image you can use :https://hub.docker.com/r/priyan5hu/builtover/tags

## Scope of Improvement

- Complete containarization of the Project to reproduce the entire pipeline using New data in some other machine or server and view experiment mlflow.ui in it.
- Several parameters related to different components in src can be added inside params.yaml.
- Storage Bucket such as AWS s3 can be used instead of dagshub.
- child runs can be added to see how different models perform using mlflow.
- We can introce Complexity to the project by doing experimentation in different stages of the pipeline.
- we can try differnt tools to accomplish same tasks.



