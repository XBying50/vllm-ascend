"""Run the actual method with the repository CPU-test bootstrap."""
import os
from pathlib import Path
import runpy
import sys

artifact = Path(__file__).resolve().parent
repo = Path(sys.argv[1]).resolve()
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["PATH"] = str(artifact / "cpu-only-bin") + os.pathsep + os.environ["PATH"]
os.chdir(repo)
sys.path.insert(0, str(repo))
import tests.ut.conftest
import tests.ut.distributed.ascend_store._mock_deps

assert not tests.ut.conftest._npu_available
sys.argv = [str(artifact / "benchmark_query_keys.py"), *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name="__main__")
