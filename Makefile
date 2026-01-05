.PHONY: all build test clean lint coverage html venv_create

all: build

# Install Rust toolchain (if not already installed)
install-rust:
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Build Rust extension in development mode
build:
	export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 && maturin develop --release

# Build Rust extension in release mode
build-release:
	maturin build --release

# Run all tests
test: test-rust test-python

# Run Rust unit tests
test-rust:
	cargo test --manifest-path=rust/featuristic-core/Cargo.toml --release
	cargo test --manifest-path=rust/featuristic-py/Cargo.toml --release

# Run Python tests
test-python:
	python3 -m pytest -v tests

# Run tests with coverage
coverage:
	coverage run -m pytest tests && \
	coverage report -m

# Lint Python code
lint:
	python3 -m pylint --rcfile=.pylintrc src/featuristic/

# Lint Rust code
lint-rust:
	cargo clippy --manifest-path=rust/featuristic-core/Cargo.toml
	cargo clippy --manifest-path=rust/featuristic-py/Cargo.toml

# Format Rust code
fmt-rust:
	cargo fmt --manifest-path=rust/featuristic-core/Cargo.toml
	cargo fmt --manifest-path=rust/featuristic-py/Cargo.toml

# Clean build artifacts
clean:
	cargo clean
	rm -rf target/
	rm -rf build docs/_build dist src/featuristic.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete

# Build documentation
html:
	rm -rf docs/_build && \
	cd docs && make html

# Create virtual environment
venv_create:
	python3 -m venv .venv && \
	source .venv/bin/activate && \
	python3 -m pip install -e .[dev]
