

# cr_pool

## Installation
We recommend to use the [`uv`](https://github.com/astral-sh/uv) package manager for installing all
python dependencies.
The following instructions will work for unix-based operating systems.
We use [`maturin`](https://github.com/PyO3/maturin) as a build tool.

```
# Create a new virtual environment and activate it
python -m venv .venv
source .venv/bin/activate

# Install maturin if not already present
uv pip install maturin

# Install this package and all its dependencies
# Prepend --uv if you are using the uv package manager
# otherwise omit this flag
python -m maturin develop -r --uv
```

Some parts have been documented under
[polinagaindrik.github.io/pool_model_paper/](https://polinagaindrik.github.io/pool_model_paper/).
