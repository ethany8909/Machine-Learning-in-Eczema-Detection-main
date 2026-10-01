import os
import tempfile

# Keep anything a test imports from writing into the repository's results/ folder.
os.environ.setdefault("DERMAFAIR_RESULTS_DIR", tempfile.mkdtemp(prefix="dermafair-test-results-"))
os.environ.setdefault("DERMAFAIR_DATA_DIR", tempfile.mkdtemp(prefix="dermafair-test-data-"))
