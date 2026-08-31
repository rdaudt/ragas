def test_ragas_imports_with_locked_dependencies() -> None:
    import ragas

    assert ragas.__version__ == "0.4.3"

