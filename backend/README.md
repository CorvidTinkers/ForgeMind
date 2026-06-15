# ForgeMind — Dataset Intelligence Layer (Phase 1)

AI-powered platform that transforms user-provided datasets into domain-specific Small Language Models (SLMs).

**Phase 1** focuses entirely on the **Dataset Intelligence Layer**: understanding datasets and generating rich metadata for future ForgeMind agents.

## Environment Architecture

**Development Environment:**
- **Host OS:** Windows 11
- **Runtime:** WSL2 Ubuntu 24.04
- **Containers:** Docker Desktop

```text
Windows 11
   ↓
WSL2 Ubuntu 24.04
   ↓
Docker Desktop
   ↓
ForgeMind (Containerized Services)
```

## Data Processing Architecture

```
User Upload
      │
      ▼
Dataset Validation Agent
      │
      ▼
Fingerprinting Engine
      │
      ▼
Dataset Registry
      │
      ▼
Dataset Analyzer Agent
      ├── Format Detector
      ├── Language Detector (FastText)
      ├── Domain Detector (Sentence Transformers)
      ├── Task Detector
      ├── Dataset Profiler
      └── Knowledge Extractor (spaCy)
      │
      ▼
Metadata Builder → PostgreSQL (JSONB)
      │
      ▼
Redis Events → Future Agents
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI |
| Database | PostgreSQL (JSONB) |
| Object Storage | MinIO (S3-compatible) |
| Task Queue | Celery + Redis |
| Event Bus | Redis Pub/Sub |
| Embeddings | Sentence Transformers |
| Language Detection | FastText |
| NLP | spaCy |
| Vector Store | FAISS |
| Experiment Tracking | MLflow (mocked) |
| Containerization | Docker Compose |

## Quick Start

### 1. Clone and Configure

```bash
cd ForgeMind
cp .env.example .env
```

### 2. Start Infrastructure

```bash
docker-compose up -d postgres redis minio mlflow
```

Wait for services to be healthy:
```bash
docker-compose ps
```

### 3. Download ML Models (First Time)

```bash
# FastText language detection model
mkdir -p backend/models
wget -O backend/models/lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

# spaCy English model
pip install spacy
python -m spacy download en_core_web_sm
```

### 4. Start the Application

```bash
docker-compose up -d api worker
```

Or run locally for development:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
celery -A app.workers.celery_app worker --loglevel=info
```

### 5. Access

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **MinIO Console**: http://localhost:9001
- **MLflow UI**: http://localhost:5000

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/datasets/upload` | Upload a dataset file (PDF, TXT, CSV, JSON) |
| `POST` | `/api/v1/datasets/analyze/{id}` | Trigger analysis workflow |
| `GET` | `/api/v1/datasets/{id}` | Get full intelligence report |
| `GET` | `/api/v1/datasets/{id}/status` | Get processing status |
| `GET` | `/api/v1/datasets/` | List all datasets |
| `GET` | `/health` | System health check |

## Example Workflow

```bash
# 1. Upload a dataset
curl -X POST http://localhost:8000/api/v1/datasets/upload \
  -F "file=@medical_qa.json"

# Response: {"dataset_id": "abc-123", "status": "uploaded"}

# 2. Trigger analysis
curl -X POST http://localhost:8000/api/v1/datasets/analyze/abc-123

# 3. Check status
curl http://localhost:8000/api/v1/datasets/abc-123/status

# 4. Get full report
curl http://localhost:8000/api/v1/datasets/abc-123
```

## Example Intelligence Report

```json
{
  "format_detection": {"format": "json", "confidence": 0.95},
  "language": {"language": "en", "confidence": 0.98},
  "top_domains": [
    {"domain": "healthcare", "score": 0.94},
    {"domain": "biology", "score": 0.82},
    {"domain": "education", "score": 0.55}
  ],
  "task_detection": {"task": "qa", "confidence": 0.91},
  "profile": {
    "sample_count": 5000,
    "avg_tokens": 120,
    "max_tokens": 512,
    "duplication_score": 0.08,
    "quality_score": 0.89
  },
  "knowledge": {
    "entities": ["Diabetes", "Insulin", "Treatment"],
    "relationships": [
      {"source": "Diabetes", "target": "Insulin", "relation": "Treatment"}
    ]
  }
}
```

## Testing

```bash
cd backend
pytest app/tests/ -v
```

## Project Structure

```
backend/app/
├── api/                  # FastAPI routes and dependencies
├── agents/               # Validation Agent, Analyzer Agent
│   └── submodules/       # Format, Language, Domain, Task, Profiler, Knowledge
├── application/          # Metadata Builder
├── domain/               # Models, Schemas, Enums
├── events/               # Redis event publisher
├── infrastructure/       # Database, FAISS, MLflow
├── services/             # Fingerprint, Registry, Parser, Agent Memory
├── storage/              # MinIO client
├── workers/              # Celery app and tasks
└── tests/                # Unit and integration tests
```

## Future Phases

This architecture is designed to support:

- **Phase 2**: Data Augmentation Agent
- **Phase 3**: Pipeline Planner Agent + Pipeline Generator
- **Phase 4**: Training Executor (SmolLM + QLoRA)
- **Phase 5**: Evaluation Agent
- **Phase 6**: Feedback Interpreter Agent + Human Review Loop
- **Phase 7**: Model Packaging & Deployment
