PYTHON ?= python
export PYTHONPATH := src

.PHONY: install lint format test train evaluate run docker all

install: ## Install runtime + dev dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

lint: ## Ruff lint + format check
	ruff check .
	ruff format --check .

format: ## Auto-fix lint issues and format code
	ruff check --fix .
	ruff format .

test: ## Run tests with coverage gate
	pytest --cov=src --cov-report=term-missing --cov-fail-under=80

train: ## Train the model -> models/model.joblib
	$(PYTHON) -m avoripe.train

evaluate: ## Evaluate + quality gate (MIN_F1) -> metrics/
	$(PYTHON) -m avoripe.evaluate

run: ## Serve the API on http://localhost:7860
	uvicorn avoripe.api:app --host 0.0.0.0 --port 7860

docker: ## Build and run the Docker image
	docker build -t avoripe:local .
	docker run --rm -p 7860:7860 avoripe:local

all: lint test train evaluate
