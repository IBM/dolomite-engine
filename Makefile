install:
	pip install -r requirements.txt
	git submodule update --init --recursive
	cd cute-kernels
	pip install .
	cd ..

install-dev:
	pip install -r requirements-dev.txt
	git submodule update --init --recursive
	cd cute-kernels
	pip install .
	cd ..

test:
	RUN_SLOW=True pytest tests

test-fast:
	RUN_SLOW=False pytest tests

update-precommit:
	pre-commit autoupdate

style:
	pre-commit run --all-files
