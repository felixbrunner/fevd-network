
install:
	pip install --upgrade pip wheel pip-tools
	pip-sync requirements/requirements.txt \
		requirements/requirements-ci.txt \
		requirements/requirements-dev.txt
	pip install -e .
	sudo apt-get install libgfortran3

requirements:
	pip install pip-tools
	pip-compile requirements/requirements.in
	pip-compile requirements/requirements-ci.in
	pip-compile requirements/requirements-dev.in

tests:
	pytest tests

format:
	isort euraculus notebooks tests
	black euraculus notebooks tests

lint:
	flake8 euraculus tests
	# mypy euraculus tests

.PHONY: install requirements tests format lint