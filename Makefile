PYTHON ?= python3
BASELINE_CONFIG ?= configs/baseline.yaml
SMOKE_CONFIG ?= configs/smoke.yaml

.PHONY: verify-data build-data validate-data run-baseline report reproduce smoke release-check test

verify-data:
	$(PYTHON) scripts/verify_data.py

build-data: verify-data
	$(PYTHON) scripts/build_model_inputs.py

validate-data: verify-data
	$(PYTHON) scripts/validate_inputs.py

run-baseline: build-data
	$(PYTHON) run_from_excel.py --config $(BASELINE_CONFIG)

report:
	$(PYTHON) scripts/generate_reports.py --results-dir results/baseline --reports-dir reports

reproduce: build-data validate-data
	$(PYTHON) run_from_excel.py --config $(BASELINE_CONFIG)
	$(PYTHON) scripts/generate_reports.py --results-dir results/baseline --reports-dir reports

smoke: build-data validate-data
	$(PYTHON) run_from_excel.py --config $(SMOKE_CONFIG)
	$(PYTHON) scripts/generate_reports.py --results-dir results/smoke --reports-dir reports/smoke

release-check:
	$(PYTHON) scripts/check_release.py

test:
	$(PYTHON) -m pytest
