# Legacy Streamlit Notes

These files are archived design notes from the old Streamlit UI era.

The product entry is now FastAPI + React:

- Frontend: `web/src/main.tsx`
- Backend: `api/main.py`
- Local app: `http://127.0.0.1:5173/`
- Local API: `http://127.0.0.1:8000/`

Do not use these notes as implementation guidance for new product work. Some
archived files also preserve historical encoding issues from the original notes;
they are kept only as context for past UI decisions.

`runtime_patches/` and `requirements-legacy.txt` are archived with the same
intent: they document the old UI runtime only and are not imported by the
FastAPI + React product path.
