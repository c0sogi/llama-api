from pathlib import Path

test_output = "bar baz"


def foo():
    print(test_output)


def test_venv():
    from llama_api.utils.venv import VirtualEnvironment

    venv_name = ".venv"
    venv = VirtualEnvironment(Path(__file__).parent / venv_name)
    assert venv.recreate() == 0, "Failed to create virtual environment."
    assert venv.pip("install", "requests") == 0, "Failed to install requests."
    stdout = venv.run_script(__file__).stdout
    assert (
        test_output in stdout
    ), "Failed to run script in virtual environment."


if __name__ == "__main__":
    foo()
