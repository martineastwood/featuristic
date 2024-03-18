.PHONY:
venv_create:
	python3 -m venv venv && \
	source venv/bin/activate && \
	python3 -m pip install .[dev]


.PHONY:
test:
	python3 -m pytest -v tests

.PHONY:
lint:
	python3 -m pylint --rcfile=.pylintrc src/numerately/

make clean:
	rm -rf build docs/_build dist src/numerately.egg-info src/numerately/__pycache__ src/numerately/*.pyc src/numerately/*/__pycache__ src/numerately/*/*.pyc

html:
	rm -rf docs/_build && \
	cd docs && make html