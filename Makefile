.PHONY: venv install translate build-sft train-demo train-final train-7b eval-demo eval-final eval-7b pipeline-demo pipeline-final

ifneq (,$(wildcard .env))
include .env
export
endif

PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

venv:
	$(PYTHON) -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install:
	$(PYTHON) -m pip install -r requirements.txt

translate:
	TARGET_LANGS=bn TRANSLATION_BACKEND=api $(PYTHON) src/03_translate.py --config configs/translation_1000_api_bn.yaml

build-sft:
	TARGET_LANGS=bn $(PYTHON) src/05_build_sft.py --config configs/translation_1000_api_bn.yaml

train-demo:
	BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct $(PYTHON) src/06_train_sft.py --config configs/training_demo.yaml

train-final:
	BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct $(PYTHON) src/06_train_sft.py --config configs/training_final.yaml

eval-demo:
	BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct $(PYTHON) src/07_eval.py --config configs/eval_demo.yaml

eval-final:
	BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct $(PYTHON) src/07_eval.py --config configs/eval_final.yaml

train-7b:
	$(PYTHON) src/06_train_sft.py --config configs/training_7b_qlora_bn.yaml

eval-7b:
	$(PYTHON) src/07_eval.py --config configs/eval_7b_qlora_bn.yaml

pipeline-demo: translate build-sft train-demo eval-demo

pipeline-final: translate build-sft train-final eval-final
