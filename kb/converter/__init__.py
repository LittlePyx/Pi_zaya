__all__ = ["PDFConverter", "ConvertConfig", "LlmConfig", "main"]


def __getattr__(name: str):
    # Lightweight coordinator imports must not initialize the full PDF/LLM stack.
    if name == "PDFConverter":
        from .pipeline import PDFConverter
        value = PDFConverter
    elif name in {"ConvertConfig", "LlmConfig"}:
        from . import config
        value = getattr(config, name)
    elif name == "main":
        from .runner import main
        value = main
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
