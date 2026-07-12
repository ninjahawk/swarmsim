"""Run one research script under the capstone figure style (see figstyle.py).

    python capstone/_shim_run.py <script.py>

Applies figstyle (rcParams + palette remap + savefig redirect into
docs/figures/) and then executes the untouched script as __main__ with the
current working directory unchanged (repo root, as the scripts expect).
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python capstone/_shim_run.py <script.py>")
    figstyle.apply()
    runpy.run_path(sys.argv[1], run_name="__main__")
