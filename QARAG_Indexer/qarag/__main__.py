"""
Entry point so the package can be run with:

    python -m qarag <command> [options]

It simply hands control to the Typer command-line app defined in cli.py.
"""
from qarag.cli import app

if __name__ == "__main__":
    app()
