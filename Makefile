.PHONY: venv install quick quick-small full demo ui ui-check clean fresh fresh-full logs translate-llm

ifneq (,$(wildcard .env))
include .env
export
endif

# Use venv Python if present, else python3 (so "make" works without activating venv)
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

venv:
	$(PYTHON) -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install:
	pip install -r requirements.txt

quick:
	$(PYTHON) src/01_download.py
	$(PYTHON) src/02_preprocess.py --config configs/preprocess_1000.yaml
	TARGET_LANGS=bn TRANSLATION_BACKEND=api $(PYTHON) src/03_translate.py --config configs/translation_1000_api_bn.yaml
	$(PYTHON) src/04_quality_checks.py --config configs/translation_1000_api_bn.yaml
	TARGET_LANGS=bn $(PYTHON) src/05_build_sft.py --config configs/translation_1000_api_bn.yaml
	BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct $(PYTHON) src/06_train_sft.py --config configs/training_1000.yaml
	BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct $(PYTHON) src/07_eval.py --config configs/eval_1000.yaml

# Same as quick but 0.5B model (fits MacBook / Colab free). Overrides BASE_MODEL so .env does not force 7B.
quick-small:
	$(PYTHON) src/01_download.py
	$(PYTHON) src/02_preprocess.py --config configs/preprocess_1000.yaml
	TARGET_LANGS=bn TRANSLATION_BACKEND=api $(PYTHON) src/03_translate.py --config configs/translation_1000_api_bn.yaml
	TARGET_LANGS=bn $(PYTHON) src/05_build_sft.py --config configs/translation_1000_api_bn.yaml
	BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct $(PYTHON) src/06_train_sft.py --config configs/training_1000.yaml
	BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct $(PYTHON) src/07_eval.py --config configs/eval_1000.yaml

full:
	bash scripts/demo.sh

demo: full

ui:
	$(PYTHON) -m streamlit run ui/app.py

ui-check:
	$(PYTHON) scripts/ui_smoke_check.py

# Remove all pipeline data (DATA_DIR, OUTPUTS_DIR, REPORTS_DIR, CACHE_DIR contents). Use before a clean re-run.
clean:
	bash scripts/fresh_run.sh clean

# Clean, then download from Hugging Face and run pipeline (quick: bn-only by default).
fresh:
	bash scripts/fresh_run.sh clean quick

# Clean, then download and run full pipeline (bn + ar + es, more steps).
fresh-full:
	bash scripts/fresh_run.sh clean full

# Step 3 only: translate with LLM (GPT). Set OPENAI_API_KEY and TRANSLATION_BACKEND=api in .env. Run 01 and 02 first.
translate-llm:
	TARGET_LANGS=bn TRANSLATION_BACKEND=api $(PYTHON) src/03_translate.py --config configs/translation_1000_api_bn.yaml

# List latest pipeline log files (REPORTS_DIR/logs). Use tail -f reports/logs/<file> to watch.
logs:
	@_dir="$${REPORTS_DIR:-./reports}/logs"; mkdir -p "$$_dir" 2>/dev/null || true; \
	ls -lt "$$_dir" 2>/dev/null | head -20 || echo "No logs yet. Run the pipeline first."
