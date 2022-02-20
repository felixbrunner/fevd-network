.PHONY: install requirements

install:
		pip install --upgrade pip wheel pip-tools
		pip-sync requirements/requirements.txt
		pip install -e .
		sudo apt-get install libgfortran3

requirements:
		pip install pip-tools
		pip-compile requirements/requirements.in