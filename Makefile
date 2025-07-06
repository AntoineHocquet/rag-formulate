# Makefile for rag-formulate

test:
	PYTHONPATH=$(shell pwd) pytest -s -v

install:
	pip install -r requirements.txt

reset:
	rm -rf venv
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

sandbox:
	PYTHONPATH=. python3 sandbox.py

# New rule for reformulation pipeline
reformulate:
	PYTHONPATH=$(PWD) python3 reformulate.py \
		--reference data/reference_texts/proust.txt \
		--input data/input/to_rewrite.txt \
		--output data/output/reformulated.txt \
		--top_k 3

# rule to recursively clean pychache and pytest cache
clean:
	find . -type d -name '__pycache__' -exec rm -r {} +
	find . -type d -name '.pytest_cache' -exec rm -r {} +