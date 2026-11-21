.PHONY: install collect process index test evaluate candidates run
install:
	python -m pip install -e ".[dev]"
collect:
	python -m modprecedent.cli collect --max-pages 5
process:
	python -m modprecedent.cli process
index:
	python -m modprecedent.cli index
test:
	python -m pytest
evaluate:
	python -m modprecedent.cli evaluate --dataset data/evaluation/approved_cases.jsonl
candidates:
	python -m modprecedent.cli candidates
run:
	python -m modprecedent.app

