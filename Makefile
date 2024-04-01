.PHONY:
venv_create:
	python3 -m venv .venv && \
	source .venv/bin/activate && \
	python3 -m pip install -e .[dev] && \
	source .venv/bin/activate


.PHONY:
test:
	python3 -m pytest -v tests/test_preprocess.py

.PHONY:
coverage:
	coverage run -m pytest tests && \
	coverage report -m

.PHONY:
lint:
	python3 -m pylint --rcfile=.pylintrc src/featuristic/

.PHONY:
make clean:
	rm -rf build docs/_build dist src/featuristic.egg-info src/featuristic/__pycache__ src/numfeaturisticerately/*.pyc src/featuristic/*/__pycache__ src/featuristic/*/*.pyc

.PHONY:
html:
	rm -rf docs/_build && \
	cd docs && make html
