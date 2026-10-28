#!/usr/bin/env bash
set -euo pipefail
python scripts/fetch_data.py            # no-op unless HF_DATASET_REPO is set
case "${MODE:-api}" in
  api) exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-7860}" ;;
  ui)  exec streamlit run app/streamlit_app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0 ;;
  all) export API_PORT=8000 API_URL="http://127.0.0.1:8000"
       uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
       exec streamlit run app/streamlit_app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 ;;
  *) echo "unknown MODE" >&2; exit 2 ;;
esac
