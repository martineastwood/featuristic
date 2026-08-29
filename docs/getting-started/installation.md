# Installation

This documentation in the `nim` branch describes **Featuristic 2.0** (compiled Nim backend). That version is **not on PyPI**.

[featuristic.co.uk](https://www.featuristic.co.uk/) and `pip install featuristic` are still **1.1.0** (pure Python, Python 3.8+).

## Support matrix (2.0 / this branch)

| Item | Status |
| --- | --- |
| CPython | 3.10–3.14 |
| Python 3.8 / 3.9 | Dropped (nuwa-build requires 3.10+) |
| OS | Linux, macOS, Windows (native architecture) |
| Linux wheels (when published) | manylinux x86_64 |
| Nim | 2.2.10 in CI (not required for a future *wheel* install) |
| nuwa-build | 0.5.1+ |
| nuwa_sdk | 0.4.4 |

Free-threaded CPython, PyPy, musllinux, and Linux aarch64 are not in the tested matrix.

---

## Public install (1.1.0)

```bash
pip install featuristic
```

That does **not** install this branch.

---

## Develop this branch

You need Nim on `PATH` (`nim --version`) and CPython 3.10+.

```bash
git clone https://github.com/martineastwood/featuristic.git
cd featuristic
git checkout nim
pip install "nuwa-build>=0.5.1"
pip install -e ".[dev]"
nuwa develop
pytest
```

`pip install .` from a source tree also needs Nim (PEP 517 backend). `pip install -e .` does not compile the extension by itself; run `nuwa develop`.

---

## Verification (after `nuwa develop`)

```python
import featuristic as ft
from featuristic import featuristic_lib

print(ft.__version__)           # 2.0.0
print(featuristic_lib.getVersion())  # 2.0.0
```

---

## Ecosystem

* **NumPy (>= 1.25.0) & Pandas (>= 2.0.0):** Data frames at the Python API; the Nim path uses Fortran-contiguous float64 internally (copied in `fit`/`transform`).
* **Scikit-Learn (>= 1.4.0):** `BaseEstimator` / `TransformerMixin`.
* **Matplotlib, tqdm, ucimlrepo:** plots, progress, example datasets.
