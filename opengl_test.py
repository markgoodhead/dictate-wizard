import os
import ctypes
import ctypes.util

cwd = os.getcwd()
for lib_name in ["opengl32", "libglapi"]:
    name = ctypes.util.find_library(lib_name)
    if name is None:
        print(f"{lib_name}.dll not found.")
    else:
        lib = ctypes.CDLL(name)
        path = ctypes.create_unicode_buffer(1024)  # adjust buffer size if necessary
        ctypes.windll.kernel32.GetModuleFileNameW(ctypes.c_void_p(lib._handle), path, len(path))
        print(f"{lib_name}.dll path: {path.value}")

    # Now check in the current working directory
    local_path = os.path.join(cwd, f"{lib_name}.dll")
    if os.path.exists(local_path):
        print(f"Local {lib_name}.dll found: {local_path}")
    else:
        print(f"Local {lib_name}.dll not found")
