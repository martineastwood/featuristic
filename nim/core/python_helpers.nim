# Python C API helpers for GIL management
# This provides direct access to Python's threading functions via nimpy

import nimpy/py_lib
import std/dynlib

# Dynamically load GIL functions from Python library (nimpy approach)
var
  PyEval_SaveThread_Fn*: proc(): pointer {.cdecl.}
  PyEval_RestoreThread_Fn*: proc(tstate: pointer) {.cdecl.}
  gilFunctionsInitialized = false

# Initialize the function pointers from nimpy's pyLib
proc initGILFunctions*() =
  ## Load GIL management functions from Python library
  ## Called automatically during module initialization
  if not gilFunctionsInitialized:
    let libHandle = py_lib.pyLib.module
    if not libHandle.isNil:
      PyEval_SaveThread_Fn = cast[typeof(PyEval_SaveThread_Fn)](symAddr(libHandle, "PyEval_SaveThread"))
      PyEval_RestoreThread_Fn = cast[typeof(PyEval_RestoreThread_Fn)](symAddr(libHandle, "PyEval_RestoreThread"))
      gilFunctionsInitialized = true

# Cython-style "with nogil" template
template withNogil*(body: untyped) =
  ## Release the Python GIL, execute body, then reacquire the GIL
  ## This allows pure Nim code to run without Python interpreter interference
  ## Equivalent to Cython's "with nogil:" block or Py_BEGIN_ALLOW_THREADS in C
  block:
    # Ensure GIL functions are initialized before use
    if not gilFunctionsInitialized:
      initGILFunctions()

    let tstate = PyEval_SaveThread_Fn()
    try:
      body
    finally:
      PyEval_RestoreThread_Fn(tstate)
