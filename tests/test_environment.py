import sys

import pytest

REQUIRED_PYTHON = "python3"


@pytest.mark.skipif(sys.version_info.major != 3, reason="Requires Python 3")
def test_environment():
    system_major = sys.version_info.major
    assert system_major in [2, 3], "Unrecognized Python version: {}".format(sys.version)

    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3

    assert (
        system_major == required_major
    ), "This project requires Python {}. Found: Python {}".format(
        required_major, sys.version
    )
    print(">>> Development environment passes all tests!")


if __name__ == "__main__":
    test_environment()
