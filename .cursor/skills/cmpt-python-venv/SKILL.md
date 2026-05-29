---
name: cmpt-python-venv
description: Runs Python, pip, and project scripts using the CMPT419 layman-scientific-embeddings virtualenv at /usr/shared/CMPT/scratch/alp11/venv/. Use when executing Python commands, installing packages, running finetune or evaluation scripts, or when the user mentions python, pip, venv, or dependencies in this repository.
---

# CMPT419 Project Python Environment

This skill applies only to the **layman-scientific-embeddings** repository.

## Default interpreter

Use this venv for all Python work in this project. Do not use system `python`, `python3`, or other venvs unless the user explicitly asks.

| Tool    | Path                                             |
| ------- | ------------------------------------------------ |
| Python  | `/usr/shared/CMPT/scratch/alp11/venv/bin/python` |
| pip     | `/usr/shared/CMPT/scratch/alp11/venv/bin/pip`    |
| Version | Python 3.12.3                                    |

## Running commands

Use the full interpreter path:

```bash
/usr/shared/CMPT/scratch/alp11/venv/bin/python script.py
/usr/shared/CMPT/scratch/alp11/venv/bin/pip install package
```

## Examples

```bash
# Run a project script
/usr/shared/CMPT/scratch/alp11/venv/bin/python finetune-vanilla-qwen.py

# Install a dependency
/usr/shared/CMPT/scratch/alp11/venv/bin/pip install torch

# One-liner module check
/usr/shared/CMPT/scratch/alp11/venv/bin/python -c "import torch; print(torch.__version__)"
```

## Notes

- Venv root: `/usr/shared/CMPT/scratch/alp11/venv/`
- If the path is missing or broken, tell the user before falling back to another Python.
