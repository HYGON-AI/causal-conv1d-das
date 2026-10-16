__version__ = "1.5.4"

import glob
import os
import sysconfig

from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_update, causal_conv1d_fn_hcu


def get_include():
    return os.path.join(os.path.dirname(__file__), "include")


def get_library_path():
    package_root = os.path.dirname(os.path.dirname(__file__))
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    exact_path = os.path.join(package_root, f"causal_conv1d_cuda{suffix}")
    if os.path.exists(exact_path):
        return exact_path
    matches = glob.glob(os.path.join(package_root, "causal_conv1d_cuda*.so"))
    if matches:
        return matches[0]
    return exact_path
