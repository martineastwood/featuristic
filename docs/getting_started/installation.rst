Installation
============

Installing Featuristic is straightforward. Choose the method that best fits your needs.

.. contents:: Table of Contents
   :local:
   :depth: 2

Install from PyPI (Recommended)
--------------------------------

The easiest way to install Featuristic is using pip:

.. code-block:: bash

   pip install featuristic

This will install the latest stable release from PyPI, including pre-compiled binaries for your platform.

**Requirements:**

* Python 3.8 or higher
* NumPy >= 1.25.0
* Pandas >= 2.0.0
* scikit-learn >= 1.4.0
* Matplotlib >= 3.0.0

Verify Installation
~~~~~~~~~~~~~~~~~~~

To verify your installation, run:

.. code-block:: python

   import featuristic
   print(featuristic.__version__)

You should see the version number printed without any errors.

Install from Source
-------------------

Installing from source allows you to access the latest development features but requires the Rust toolchain.

Prerequisites
~~~~~~~~~~~~~

**Rust Toolchain:**

Featuristic uses Rust for performance-critical code (5-20x speedup). You need to install Rust first:

.. code-block:: bash

   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

Or visit https://rustup.rs/ for platform-specific instructions.

**Build Dependencies:**

.. code-block:: bash

   # Install maturin (Rust-Python build tool)
   pip install maturin

Installation Steps
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/martineastwood/featuristic.git
   cd featuristic

   # Build and install in development mode
   maturin develop --release

The ``--release`` flag enables optimizations for production use. Omit it for faster compilation during development.

Install from Pre-built Wheels
------------------------------

Pre-built wheels are available for common platforms (Linux, macOS, Windows) and Python versions. These are installed automatically when using ``pip install featuristic``.

If you need a specific version or platform:

.. code-block:: bash

   pip install featuristic==2.0.0

Development Installation
-------------------------

If you want to contribute to Featuristic or modify the source code:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/martineastwood/featuristic.git
   cd featuristic

   # Create a virtual environment (recommended)
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install in development mode with all dev dependencies
   pip install -e ".[dev]"

This installs:

* The package in editable mode
* Development tools (black, pylint, pytest, etc.)
* Documentation tools (Sphinx, nbsphinx, pydata-sphinx_theme)

Building the Rust Extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After modifying Rust code, you need to rebuild:

.. code-block:: bash

   # Development build (faster compilation)
   maturin develop

   # Release build (optimized, slower compilation)
   maturin develop --release

Or use the provided Makefile:

.. code-block:: bash

   make build

Docker Installation
-------------------

For containerized environments, use the provided Dockerfile:

.. code-block:: bash

   # Build the image
   docker build -t featuristic .

   # Run in a container
   docker run -it featuristic

Troubleshooting
---------------

Rust Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``cargo not found``

**Solution:** Make sure Rust is installed and in your PATH:

.. code-block:: bash

   # Check Rust installation
   cargo --version
   rustc --version

   # If not found, reinstall or add to PATH
   export PATH="$HOME/.cargo/bin:$PATH"

Build Failures
~~~~~~~~~~~~~~

**Problem:** Compilation fails with ``linking with cc failed``

**Solution:** Install system-level build dependencies:

**macOS:**

.. code-block:: bash

   xcode-select --install

**Ubuntu/Debian:**

.. code-block:: bash

   sudo apt-get install build-essential

**Fedora/RHEL:**

.. code-block:: bash

   sudo dnf install gcc gcc-c++

Python Version Issues
~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``Python version not supported``

**Solution:** Featuristic requires Python 3.8+. Check your version:

.. code-block:: bash

   python --version

   # If using Python 2.x, use python3 instead
   python3 -m pip install featuristic

Import Errors
~~~~~~~~~~~~~

**Problem:** ``ImportError: dynamic module does not define init function``

**Solution:** The Rust extension wasn't built properly. Rebuild from source:

.. code-block:: bash

   pip uninstall featuristic
   cd featuristic
   maturin develop --release

Permission Errors
~~~~~~~~~~~~~~~~~

**Problem:** ``Permission denied`` when installing

**Solution:** Use a virtual environment (recommended) or user installation:

.. code-block:: bash

   # Option 1: Use virtual environment (RECOMMENDED)
   python -m venv .venv
   source .venv/bin/activate
   pip install featuristic

   # Option 2: User installation (alternative)
   pip install --user featuristic

Platform-Specific Notes
-----------------------

Apple Silicon (M1/M2/M3)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Apple Silicon Macs work perfectly. Just ensure you have:

.. code-block:: bash

   # Install Rust for ARM64
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

   # Install normally
   pip install featuristic

Windows
~~~~~~~

**Requirements:**

* Microsoft Visual C++ 14.0 or greater (for Rust)
* Rust installer from https://rustup.rs/

**Installation:**

.. code-block:: powershell

   # Install Rust
   # Download and run rustup-init.exe from https://rustup.rs/

   # Install Featuristic
   pip install featuristic

Linux
~~~~~

Most Linux distributions work out of the box. Ensure you have:

.. code-block:: bash

   # Debian/Ubuntu
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip build-essential

   # Fedora/RHEL
   sudo dnf install python3-devel python3-pip gcc gcc-c++

   # Install Featuristic
   pip3 install featuristic

Next Steps
----------

Now that you have Featuristic installed, check out:

* :doc:`quickstart` - Get started in 5 minutes
* :doc:`next_steps` - Choose your learning path
* :doc:`../user_guide/feature_synthesis` - Deep dive into feature synthesis
* :doc:`../user_guide/feature_selection` - Evolutionary feature selection

Still having issues? `Check the GitHub issues <https://github.com/martineastwood/featuristic/issues>`_ or `open a new issue <https://github.com/martineastwood/featuristic/issues/new>`_.
