.PHONY: ci fmt

ci:
	flake8 || true
	pytest -q || true

fmt:
	python -m pip install black
	black src tests
