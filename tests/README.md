# Test Suite

Automated tests live under this directory and are the only paths collected by
`pytest.ini`.

- `unit/`: fast module and contract tests.
- `sanity/`: lightweight integration and startup checks.

Run the full Python suite from the repository root:

```powershell
python -m pytest tests/
```

The product runtime is FastAPI + React. Legacy Streamlit entrypoints and old
root-level manual test scripts have been removed from `main`.
