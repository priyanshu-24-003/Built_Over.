# Built_Over 🐧➜🔧

About
Built_Over. is a derivative Repository of forked repo palmer-penguins-classification ,This Repo uses MLOPs concepts OOPs, data versioning, experiment tracking, continuous Integration and few more.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MLOps](https://img.shields.io/badge/MLOps-DVC%20%7C%20MLflow%20%7C%20DagsHub%20%7C%20GitHub%20Actions-important)](https://mlops.community/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Purpose

**Built_Over** is a **heavily enhanced fork** of a classic Palmer Penguins species classification project.

The goal was **not** just building another model — but demonstrating how to use **Mlops Concepts** using:

- Used **object-oriented** code structure
- **Data versioning** (hard data + pipeline)
- **Experiment tracking** & artifact logging
- **Remote experiment storage** 
- **Continuous Integration** (automated testing), 
- **Containerization** readiness

All while keeping the beloved **Palmer Penguins** dataset (Adelie, Chinstrap, Gentoo 🐧).

## ✨ Key Features & Modern MLOps Practices Demonstrated

| Area                        | Tool/Technique                          | What was implemented                                                                 |
|-----------------------------|-----------------------------------------|--------------------------------------------------------------------------------------|
| 📂 Data & Pipeline Versioning | **DVC**                                | Versioned raw & processed data, automated reproducible pipelines (`dvc repro`)      |
| 📦 Remote Data & Model Store  | **DagsHub**                            | DVC remote storage + MLflow remote tracking server                                  |
| 📊 Experiment Tracking       | **MLflow**                             | Logged metrics, parameters, models, datasets, figures — viewable in MLflow UI       |
| 🧪 Continuous Integration    | **GitHub Actions**                     | Automated tests for components, pipeline validation, logger checks                  |
| 🐳 Containerization          | **Docker**                             | Dockerfile + image build setup (ready for future deployment)                        |
| 🛠 Clean Code Architecture   | **OOP principles**                     | Modular helpers: `Loader`, `Saver`, `Logster` + inheritance usage                   |
| 🔄 Reproducibility           | DVC + `dvc.lock` + dvc repro              | Commit after every meaningful pipeline change — fast local runs without `dvc repro` |
