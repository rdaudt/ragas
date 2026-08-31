from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_loading_credentials() -> None:
    app_path = Path(__file__).parents[1] / "app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "RAGAS Evaluation Demo"
    assert app.chat_input[0].placeholder == "Ask a question about the indexed documents"

