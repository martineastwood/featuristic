.PHONY:
venv_create:
	python3 -m venv venv
	python3 -m pip install .[dev]


.PHONY:
test:
	python3 -m pytest -v tests

.PHONY:
lint:
	python3 -m pylint --rcfile=.pylintrc ./src/featurize/selection/mrmr.py