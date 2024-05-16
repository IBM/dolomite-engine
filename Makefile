install:
	pip install .

install-dev:
	pip install -e .

test:
	pytest tests

update-precommit:
	pre-commit autoupdate

style:
	pre-commit run --all-files
