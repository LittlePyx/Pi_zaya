
import sys
import os
import pytest

# Add project root to sys.path so we can import 'kb'
# This assumes tests/conftest.py is located at <root>/tests/conftest.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
