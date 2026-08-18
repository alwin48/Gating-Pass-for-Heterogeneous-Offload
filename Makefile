.PHONY: pass dataset train demo all clean

PY ?= .venv/bin/python
OPT ?= /usr/lib/llvm-18/bin/opt

pass:
	./llvm-pass/build.sh

dataset:
	$(PY) ml/generate_dataset.py -n 100
	$(PY) ml/build_trace_db.py
	$(PY) schema/validate_dataset.py data/dataset.jsonl

train:
	$(PY) ml/train_eval.py
	$(PY) ml/plot_eval.py

demo:
	./scripts/demo.sh

all:
	./scripts/run_all.sh

clean:
	rm -f llvm-pass/GPHOPass.so
	rm -rf ml/artifacts
