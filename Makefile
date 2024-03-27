install:
	pip install .

install-dev:
	pip install -e .

test:
	pytest tests

style:
	pre-commit run --all-files
