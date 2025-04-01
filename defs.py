from pathlib import Path


def project_root() -> Path:
    Path(__file__).parent.mkdir(exist_ok=True, parents=True)
    return Path(__file__).parent
