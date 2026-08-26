import os
import sys

# The app modules (database, models, app, auth, ...) live at the project
# root, one level up from tests/. The test scripts import them as
# top-level modules (e.g. `from database import get_db`), so make sure
# the project root is on sys.path regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
