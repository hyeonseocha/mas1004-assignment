# Notes for coding agents

## Running Python in this repository

Run every Python command through uv, from the repository folder:

```
uv run pytest tests/test_data.py
uv run python src/collect.py --classes "espresso cup,wine glass,paper coffee cup" --n 150
uv run python src/run.py
```

uv reads `pyproject.toml` and `uv.lock`, downloads Python 3.12 itself, and
installs the exact version of every package into `.venv`. The first run takes a
few minutes. Do not activate `.venv` or call its Python directly.

- Do not use `pip install`, `python -m venv`, conda, or the `python` or
  `python3` that is already on the computer.
- Do not change versions in `pyproject.toml`, `uv.lock` or `requirements.txt`,
  and do not run `uv add`, `uv remove` or `uv lock --upgrade`. `ddgs` and
  `primp` in particular are pinned on purpose: other versions crash on macOS or
  fail to search, and `src/collect.py` stops if it finds any other version.

If `uv` is not found, install it with the official installer, then close the
terminal and quit and reopen VS Code so the new `PATH` is picked up:

- macOS or Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows, in PowerShell:
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

If uv stops with one of these errors, stop and tell the student. Do not try to
work around it with pip or other versions.

- `The current Python platform is not compatible with the lockfile's supported
  environments`: this is an Intel Mac or a Windows computer with an ARM
  processor. PyTorch has no packages for them. Use Google Colab with
  `assignment1_colab.ipynb`.
- `can't be installed because it doesn't have a source distribution or wheel
  for the current platform` on a Mac: macOS is older than 14 (Sonoma). Update
  macOS, or use Google Colab.

## On Google Colab

uv is not used on Colab. `assignment1_colab.ipynb` installs with
`pip install -q -r requirements.txt` and runs `python ...` directly.
