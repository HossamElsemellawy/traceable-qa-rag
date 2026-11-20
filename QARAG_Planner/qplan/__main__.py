"""
Entry point for `python -m qplan ...`.

Python runs this file when the package is executed as a module. It only
forwards to the command-line interface in cli.py.
"""

from qplan.cli import main

if __name__ == "__main__":
    main()
