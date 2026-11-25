PYTHON ?= python

.PHONY: test check
test:
	$(PYTHON) -m pytest

check:
	$(PYTHON) -m compileall -q src examples scripts
	$(PYTHON) scripts/check_release.py
