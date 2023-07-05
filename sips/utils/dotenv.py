import ast
import os
from pathlib import Path
from warnings import warn


def _get_file_parent_dir(src: str) -> Path:
    """
    Get the path of the desired parent of this __file__ path.

    """
    if src not in __file__:
        raise FileNotFoundError(f"'{src}' is no parent directory of '{__file__}'")
    root_dir = __file__.rsplit(f"/{src}/", 1)[0]
    return Path(root_dir)


def load_dotenv(filename: str = ".env") -> None:
    """
    Load dotenv file into 'os.environ'.

    Parameters
    ----------
    filename : str, optional
        Filename of the dotenv file
        Default: ".env"

    """
    root_dir = _get_file_parent_dir("sips")
    dotenv_path = root_dir / filename
    # Skip if file not exists
    if not dotenv_path.exists():
        return
    # Read file and write in os environment
    with open(dotenv_path) as f:
        for line in f:
            match line.strip().split("="):
                case [empty] if empty.strip() == "":  # ignore newlines
                    continue
                case alist if alist[0].startswith("#"):  # ignore comments
                    continue
                case [key, value]:
                    try:
                        # Handle when string values are in quotes
                        os.environ[key] = ast.literal_eval(value)
                    except (TypeError, ValueError):
                        os.environ[key] = value
                case _:
                    warn(f"Invalid line for {load_dotenv.__name__}: '{line.strip()}'")
