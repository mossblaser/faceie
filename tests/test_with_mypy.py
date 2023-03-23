import subprocess

from pathlib import Path

import faceie


def test_mypy():
    subprocess.run(
        [
            "mypy",
            str(Path(faceie.__file__).parent),
            str(Path(__file__).parent),
        ],
        check=True,
    )
