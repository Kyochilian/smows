import os
import sys
from importlib.machinery import PathFinder


def _try_load_real_torchtext() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    original_sys_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if os.path.abspath(p) != project_root]
        spec = PathFinder.find_spec("torchtext")
        if spec is None or spec.loader is None:
            return
        spec.loader.exec_module(sys.modules[__name__])
    finally:
        sys.path = original_sys_path


try:
    _try_load_real_torchtext()
except Exception:
    pass

