"""
Lightweight alternative for the module python-dotenv.

"""
import os
import re
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
    # Define RegEx pattern for matching environment variables
    left_pattern = r"(^[a-zA-Z0-9_]+)"
    right_sub_patterns = [r"'[^']*'", r"\"[^\"]*\"", r"[^\"'\s#]+[^\s#]*"]
    right_pattern = r"(" + r"|".join(right_sub_patterns) + r")"
    pattern = left_pattern + r"(\=)" + right_pattern
    # Read file and write in os environment
    with open(dotenv_path) as file:
        for line in file:
            m = re.search(pattern, line)
            if not m:
                if not re.match(r"\s*(#.*|$)", line):  # empty line or just comment
                    warn(f"Invalid line for {load_dotenv.__name__}: '{line.strip()}'")
                continue
            key, value = map(str.strip, m.group().split("="))
            if value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ[key] = value
