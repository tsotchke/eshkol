.PHONY: help eshkol-swarm-status eshkol-swarm-cycle eshkol-icc-audit eshkol-icc-smoke

help:
	@echo "Eshkol swarm + ICC make targets"
	@echo
	@printf '  %-30s %s\n' 'eshkol-swarm-status'    'Refresh .swarm/status.md from the task ledger'
	@printf '  %-30s %s\n' 'eshkol-swarm-cycle'     'Run scripts/run_icc_smoke.sh, refresh status, audit v1.3-release'
	@printf '  %-30s %s\n' 'eshkol-icc-smoke'       'Run the ICC smoke probes (writes scripts/icc_traces/eshkol_smoke.jsonl)'
	@printf '  %-30s %s\n' 'eshkol-icc-audit'       'Run the ICC production-audit for v1.3-release'
	@echo
	@echo "For full build options see CMakeLists.txt; this Makefile is the swarm-coordination top-level only."

eshkol-swarm-status:
	python3 scripts/eshkol_swarm_status.py --write-status

eshkol-swarm-cycle: eshkol-icc-smoke eshkol-swarm-status eshkol-icc-audit

eshkol-icc-smoke:
	bash scripts/run_icc_smoke.sh

eshkol-icc-audit:
	python3 ~/Desktop/infinite_context_coder/scripts/codebase_tool.py production-audit \
	    --repo eshkol_lang --target v1.3-release \
	    --trace-dir scripts/icc_traces --format markdown
