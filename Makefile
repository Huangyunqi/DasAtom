# Define variables
VENV_NAME = .venv
REQUIREMENTS = requirements.txt
EXAMPLE_SCRIPT = examples/example_script.py
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip

# Default target
all: run

# Create and set up the virtual environment
$(VENV_NAME)/bin/activate: $(REQUIREMENTS)
	@echo "Creating virtual environment..."
	python -m venv $(VENV_NAME)
	@echo "Activating virtual environment and installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)

# Run the example script
run: $(VENV_NAME)/bin/activate
	@echo "Running example script..."
	$(PYTHON) $(EXAMPLE_SCRIPT)

# Ensure dependencies are installed
check_dependencies: $(VENV_NAME)/bin/activate
	@$(PIP) install -r $(REQUIREMENTS)


