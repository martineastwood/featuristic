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
	python3 -m pylint --rcfile=.pylintrc src/featuring/

make clean:
	rm -rf build docs/_build dist src/featuring.egg-info src/featuring/__pycache__ src/featuring/*.pyc src/featuring/*/__pycache__ src/featuring/*/*.pyc

html:
	rm -rf docs/_build && \
	cd docs && make html