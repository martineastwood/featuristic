.PHONY:
venv_create:
	python3 -m venv .venv && \
	source .venv/bin/activate && \
	python3 -m pip install -e .[dev] && \
	source .venv/bin/activate


.PHONY:
test:
	python3 -m pytest -v tests

.PHONY:
lint:
	python3 -m pylint --rcfile=.pylintrc src/featuristic/

make clean:
	rm -rf build docs/_build dist src/featuristic.egg-info src/featuristic/__pycache__ src/featuristic/*.pyc src/featuristic/*/__pycache__ src/featuristic/*/*.pyc

html:
	rm -rf docs/_build && \
	cd docs && make html