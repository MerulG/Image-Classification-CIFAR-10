# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CIFAR-10 image classification project using PyTorch. The project is in early development — no training scripts exist yet.

- **Python version**: 3.11.7 (see `.python-version`)
- **Virtual environment**: `venv/`

## Environment Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Key Dependencies

| Purpose | Library |
|---|---|
| Deep learning | PyTorch 2.11, torchvision 0.26 |
| Experiment tracking | MLflow 3.10 |
| Data science | numpy, pandas, scikit-learn, scipy |
| Visualization | matplotlib, seaborn |
| Notebooks | JupyterLab 4.5 |
| Serving | Flask, FastAPI, Uvicorn |

## Git Commits

Never add a `Co-Authored-By: Claude` trailer to commit messages. Keep commits authored solely by the user's configured Git identity.

## Architecture Notes

- Use `torchvision.datasets.CIFAR10` for data loading
- Model definitions should use `torch.nn.Module`
- MLflow is available for experiment tracking (`mlflow.start_run()`, logging metrics/params/artifacts)
- `python-dotenv` is available for environment-based configuration (`.env` files)
- Both Flask and FastAPI are available for serving trained models
