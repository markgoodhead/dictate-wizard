import ctypes
import ctypes.util

name = ctypes.util.find_library("opengl32")
if name is None:
    print("Library not found.")
else:
    lib = ctypes.CDLL(name)
    path = ctypes.create_unicode_buffer(1024)  # adjust buffer size if necessary
    ctypes.windll.kernel32.GetModuleFileNameW(ctypes.c_void_p(lib._handle), path, len(path))
    print(path.value)
