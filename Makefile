install:
	pip install -r requirements.txt
	pip install .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest tests

update-precommit:
	pre-commit autoupdate

style:
	pre-commit run --all-files
