# Installation

## Install from PyPI

```bash
python -m pip install featuristic
```

Published wheels contain the compiled backend; users do not need to install Nim.

## Support matrix

| Item | Status |
| --- | --- |
| CPython | 3.10–3.14 |
| Python 3.8 / 3.9 | Not supported |
| OS | Linux, macOS, Windows |
| Linux wheels | manylinux x86_64 and aarch64 |
| macOS wheels | Apple Silicon (arm64) and Intel (x86_64) |
| Windows wheels | x86_64 |
| Nim | 2.2.10 for source builds; not required for wheel installs |
| nuwa-build | 0.5.3+ for source builds |
| nuwa_sdk | 0.4.4 |

Free-threaded CPython, PyPy, and musllinux are not in the tested matrix.

## Development installation

You need Nim on `PATH` (`nim --version`) and CPython 3.10+.

```bash
git clone https://github.com/martineastwood/featuristic.git
cd featuristic
pip install "nuwa-build>=0.5.3"
pip install -e ".[dev]"
nuwa develop
pytest
```

`pip install .` from a source tree also needs Nim (PEP 517 backend). `pip install -e .` does not compile the extension by itself; run `nuwa develop`.

---

## Verify the installation

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
