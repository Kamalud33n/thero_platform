"""
pytest config for the tests/ folder.

These test scripts (test_isolation.py, test_two_tab_flow.py, etc.) import
project modules directly, e.g. `from database import get_db`. They already
insert the project root into sys.path themselves when run standalone
(`python tests/test_isolation.py`), but this conftest.py makes the same
thing work automatically if the suite is ever collected with `pytest`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
