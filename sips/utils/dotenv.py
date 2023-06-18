from pathlib import Path

from dotenv import load_dotenv as _load_dotenv


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
    if dotenv_path.exists():
        _load_dotenv(dotenv_path)
