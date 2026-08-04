.PHONY: reproduce smoke test validate

reproduce:
	python scripts/reproduce_paper_results.py

smoke:
	python scripts/reproduce_paper_results.py --quick

test:
	PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider

validate:
	python scripts/validate_release.py
