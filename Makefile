install:
	pip install --extra-index-url https://download.pytorch.org/whl/nightly/cpu .

install-dev:
	pip install --extra-index-url https://download.pytorch.org/whl/nightly/cpu -e .
	pip install -r requirements-dev.txt
	
	git clone https://github.com/sustcsonglin/flash-linear-attention
	cd flash-linear-attention
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
