.PHONY: install install-ml test lint smoke pipeline pipeline-baseline api ui eval eval-answers validate docker-build docker-up

install:            ## API/agent/tests only (no torch)
	pip install -r requirements-dev.txt
install-ml:         ## full stack incl. FinBERT/embeddings (CPU)
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt -r requirements-dev.txt
test:
	python -m pytest
lint:
	ruff check .
smoke:              ## retrieval regression gate (what CI runs)
	python -m eval.smoke
validate:           ## corpus sanity checks
	python scripts/validate_corpus.py
pipeline:           ## ingest + sentiment + v2 index
	python scripts/run_pipeline.py
pipeline-baseline:  ## also build the v1 index for the A0 baseline
	python scripts/run_pipeline.py --configs v1 v2
api:
	uvicorn src.api.main:app --reload --port 7860
ui:
	streamlit run app/streamlit_app.py
eval:               ## reproduce the retrieval table (A0..A5)
	python -m eval.verify_evidence && python -m eval.run_eval retrieval
eval-answers:       ## reproduce the answer-quality table (B0..B2); needs Ollama running
	python -m eval.run_eval answers
docker-build:
	docker build -t earnings-ai .
docker-up:
	docker compose up --build
