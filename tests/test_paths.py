"""Data and results locations come from environment variables, with repo-local defaults."""

import os
import subprocess
import sys

PROBE = "import dermafair.paths as p; print(p.DATA_DIR); print(p.RESULTS_DIR); print(p.output_dir('probe'))"


def _run(env_overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith("DERMAFAIR_")}
    env.update(env_overrides)
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def test_environment_overrides(tmp_path):
    data, results = tmp_path / "data", tmp_path / "results"
    data_dir, results_dir, probe = _run({"DERMAFAIR_DATA_DIR": str(data), "DERMAFAIR_RESULTS_DIR": str(results)})
    assert data_dir == str(data.resolve())
    assert results_dir == str(results.resolve())
    assert (results / "probe").is_dir()
    assert probe == str((results / "probe").resolve())


def test_defaults_live_inside_the_repository(tmp_path):
    data_dir, results_dir, _ = _run({"DERMAFAIR_RESULTS_DIR": str(tmp_path)})
    assert data_dir.endswith("data")
    assert os.path.dirname(data_dir) == os.path.dirname(
        os.path.dirname(os.path.abspath(__import__("dermafair").__file__))
    )
