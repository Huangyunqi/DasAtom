# Makefile

.PHONY: all qft tetris

# Activate the virtual environment and run the Python scripts
VENV_ACTIVATE = . .venv/bin/activate

all: qft tetris

qft:
	$(VENV_ACTIVATE) && nohup python -u DasAtom.py qft Data/qiskit-bench/qft/qft_small > qft.log 2>&1 &

tetris:
	$(VENV_ACTIVATE) && nohup python -u DasAtom.py tetris Data/Q_Tetris > test.log 2>&1 &
