# SafeLens — Production ML Engineering Instructions

## 0. PROJECT MISSION

Build SafeLens as a production-style Trust & Safety ML system.

This is NOT a toy ML project, Kaggle notebook, chatbot, RAG demo, or social-media clone.

The objective is to demonstrate how a strong ML engineer would design, train, evaluate, deploy, monitor, improve, and roll back a multimodal content-understanding and moderation system.

The system should demonstrate:

- NLP content understanding
- Computer vision
- Multimodal representation learning
- Content-risk classification
- Semantic retrieval
- Lexical retrieval
- Ranking and reranking
- Multi-stage moderation
- Human-in-the-loop review
- Online inference
- Model versioning
- Dataset versioning
- Offline evaluation
- Regression testing
- Model deployment gates
- Shadow deployment
- Canary deployment
- Rollback
- Model/data monitoring
- Drift detection
- Feedback-driven retraining
- Production-style testing
- Reproducible ML workflows

The project should map strongly to ML engineering requirements commonly found in large-scale Trust & Safety organizations:

- multimodal content classifiers
- representation learning
- LLM/NLP/CV
- ranking
- retrieval
- online ML systems
- model serving
- data-driven strategy
- production engineering

Do NOT claim this is TikTok infrastructure or TikTok-scale.

This is a local production-style prototype inspired by the engineering patterns used in large-scale ML systems.

--------------------------------------------------
1. CORE ENGINEERING PRINCIPLE
--------------------------------------------------

Treat the MODEL as only one component of the system.

The complete lifecycle is:

DATA
  ↓
DATA VALIDATION
  ↓
DATA VERSIONING
  ↓
LABELING / HUMAN REVIEW
  ↓
TRAINING
  ↓
OFFLINE EVALUATION
  ↓
REGRESSION TESTING
  ↓
MODEL REGISTRY
  ↓
STAGING
  ↓
SHADOW
  ↓
CANARY
  ↓
PRODUCTION
  ↓
MONITORING
  ↓
HUMAN FEEDBACK
  ↓
HARD-EXAMPLE MINING
  ↓
RETRAINING

Every stage must be represented in the repository, even if some production mechanisms are simulated locally.

Do not jump directly from train.py to production.

--------------------------------------------------
2. HARDWARE
--------------------------------------------------

Primary local development machine:

Apple MacBook Air M2, 2022.

Secondary training environment:

Google Colab GPU.

LOCAL M2 must support:

- CPU
- Apple MPS when available
- inference
- FastAPI
- retrieval
- databases
- Docker
- monitoring
- benchmarking
- integration testing

COLAB GPU should be used for:

- transformer training
- multimodal training
- expensive embedding generation
- model experiments

Never assume CUDA exists locally.

Implement device detection:

1. CUDA if available
2. MPS if available
3. CPU fallback

Training scripts must work in both environments.

Do not hard-code CUDA.

--------------------------------------------------
3. TECHNOLOGY STACK
--------------------------------------------------

Use the following technologies where they provide real value.

### ML

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- DeBERTa or equivalent transformer
- CLIP or equivalent vision-language encoder
- Sentence Transformers

### Retrieval

- FAISS
- BM25
- Qdrant
- Cross-Encoder reranking

### Data

- Polars or Pandas
- PostgreSQL
- dataset manifests/versioning

### ML lifecycle

- MLflow
- Git
- GitHub Actions

DVC may be used if it materially improves dataset versioning.

### Serving

- FastAPI
- Pydantic
- Redis
- ONNX Runtime

### Infrastructure

- Docker
- Docker Compose

### Observability

- Prometheus
- Grafana
- structured logging

### Testing

- pytest

### Optional later

- Kafka
- AWS S3
- AWS EC2
- AWS ECR

Do NOT add optional technologies until the core system works.

Do NOT use:

- LangChain without a specific technical reason
- LangGraph
- Kubernetes unless explicitly required
- unnecessary microservices
- unnecessary frontend frameworks

Technology must serve the architecture.

Never add a technology merely to make the README longer.

--------------------------------------------------
4. PRODUCTION ARCHITECTURE
--------------------------------------------------

Target architecture:

                    CONTENT EVENT
                         |
                         v
                 INGESTION API
                         |
                         v
                CONTENT METADATA
                  + BLOB STORAGE
                         |
                         v
               DATA VALIDATION
                         |
              +----------+----------+
              |                     |
              v                     v
        TEXT PIPELINE         IMAGE PIPELINE
              |                     |
          DeBERTa                 CLIP/ViT
              |                     |
              +----------+----------+
                         |
                         v
                REPRESENTATIONS
                         |
                         v
               MULTIMODAL MODEL
                         |
                         v
                  POLICY SCORES
                         |
              +----------+----------+
              |                     |
              v                     v
          RISK ENGINE          RETRIEVAL
                                  |
                       +----------+----------+
                       |                     |
                      BM25                 Dense
                       |                     |
                       +----------+----------+
                                  |
                                  v
                            TOP-K CANDIDATES
                                  |
                                  v
                              RERANKER
                                  |
                                  v
                          MODERATION EVIDENCE
                                  |
                         +--------+--------+
                         |        |        |
                         v        v        v
                       ALLOW    REVIEW    BLOCK
                                  |
                                  v
                            HUMAN REVIEW
                                  |
                                  v
                             FINAL LABEL
                                  |
                                  v
                            FEEDBACK STORE
                                  |
                                  v
                         HARD EXAMPLE MINING
                                  |
                                  v
                              RETRAINING


MODEL LIFECYCLE:

Candidate
   ↓
Offline evaluation
   ↓
Quality gate
   ↓
Staging
   ↓
Shadow
   ↓
Canary
   ↓
Production
   ↓
Monitoring
   ↓
Rollback if necessary

--------------------------------------------------
5. REPOSITORY STRUCTURE
--------------------------------------------------

Use this structure unless a strong engineering reason requires otherwise:

safelens/

├── CLAUDE.md
├── README.md
├── LICENSE
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── training.yaml
│   ├── serving.yaml
│   └── monitoring.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── manifests/
│   └── README.md
│
├── src/
│   └── safelens/
│       ├── data/
│       │   ├── ingestion/
│       │   ├── validation/
│       │   └── preprocessing/
│       │
│       ├── models/
│       │   ├── text/
│       │   ├── vision/
│       │   └── multimodal/
│       │
│       ├── embeddings/
│       │
│       ├── retrieval/
│       │   ├── bm25/
│       │   ├── dense/
│       │   └── hybrid/
│       │
│       ├── ranking/
│       │
│       ├── moderation/
│       │   ├── policies/
│       │   ├── thresholds/
│       │   └── decision_engine/
│       │
│       ├── feedback/
│       │   ├── labeling/
│       │   └── hard_examples/
│       │
│       ├── serving/
│       │   ├── api/
│       │   ├── inference/
│       │   └── routing/
│       │
│       ├── monitoring/
│       │   ├── data_quality/
│       │   ├── model_quality/
│       │   ├── drift/
│       │   └── serving/
│       │
│       └── utils/
│
├── training/
│   ├── train_text.py
│   ├── train_vision.py
│   ├── train_multimodal.py
│   └── evaluate.py
│
├── evaluation/
│   ├── offline/
│   ├── regression/
│   ├── policy/
│   └── benchmarks/
│
├── deployment/
│   ├── staging/
│   ├── shadow/
│   ├── canary/
│   └── rollback/
│
├── monitoring/
│   ├── prometheus/
│   └── grafana/
│
├── scripts/
│   ├── prepare_data.py
│   ├── validate_data.py
│   ├── build_index.py
│   ├── benchmark.py
│   └── export_onnx.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── model/
│   └── serving/
│
├── docs/
│   ├── problem.md
│   ├── policy_taxonomy.md
│   ├── architecture.md
│   ├── data_contract.md
│   ├── data_pipeline.md
│   ├── model_design.md
│   ├── evaluation.md
│   ├── retrieval_design.md
│   ├── serving_design.md
│   ├── deployment.md
│   ├── monitoring.md
│   ├── incident_playbook.md
│   ├── model_card.md
│   └── experiments.md
│
├── benchmarks/
│   └── results/
│
└── notebooks/

--------------------------------------------------
6. DATA ENGINEERING
--------------------------------------------------

Do not start by training a model.

First build a reproducible data pipeline:

raw data
  ↓
schema validation
  ↓
quality validation
  ↓
deduplication
  ↓
label validation
  ↓
PII/sensitive-data checks where applicable
  ↓
split
  ↓
versioned dataset
  ↓
training

Track:

- dataset version
- source
- license
- dataset hash
- number of records
- label distribution
- duplicate count
- missing values
- split strategy
- preprocessing version
- creation timestamp

Never fabricate dataset statistics.

Every reported number must come from an actual script.

--------------------------------------------------
7. DATA CONTRACT
--------------------------------------------------

Define an explicit content schema.

Example conceptual schema:

content_id
text
image_reference
timestamp
source
label
policy_id
label_source
model_version

Use Pydantic or equivalent validation.

Invalid records should fail clearly.

Do not silently corrupt or coerce bad data.

--------------------------------------------------
8. DATA LEAKAGE
--------------------------------------------------

Explicitly test for:

- duplicate records across train/test
- near duplicates where practical
- same source content across splits
- temporal leakage
- label leakage

Document findings.

Create tests where practical.

--------------------------------------------------
9. DATA SPLIT
--------------------------------------------------

Use two evaluation strategies where data permits.

### Random split

For baseline comparison.

### Time-based split

Train on earlier data and evaluate on later data.

The time-based evaluation should simulate production distribution shift.

Document why both are useful.

--------------------------------------------------
10. POLICY TAXONOMY
--------------------------------------------------

Create a small, defensible policy taxonomy.

Potential categories:

SAFE
HARASSMENT
HATEFUL
VIOLENCE
SEXUAL
SELF_HARM
SPAM

IMPORTANT:

Only use categories actually supported by the selected public datasets.

Do not invent labels.

If the dataset only supports a subset, document the limitation.

For each policy document:

- definition
- examples
- ambiguity
- model label
- moderation action
- review requirement

--------------------------------------------------
11. LABELING / HUMAN REVIEW
--------------------------------------------------

Treat labels as data with provenance.

Store:

content_id
policy_id
label
label_source
annotator/reviewer identifier where appropriate
confidence
timestamp
model_version
review_status

Design:

Model
 ↓
high confidence → automated decision

uncertain
 ↓
human review
 ↓
final label
 ↓
feedback store

Do not assume the model is always correct.

--------------------------------------------------
12. MODEL BASELINE
--------------------------------------------------

Before transformers, implement:

TF-IDF
+
Logistic Regression

Measure:

- accuracy
- precision
- recall
- macro F1
- PR-AUC
- confusion matrix
- false-positive rate
- false-negative rate

Store results in:

benchmarks/results/

Do not delete negative results.

--------------------------------------------------
13. TEXT MODEL
--------------------------------------------------

Start with DeBERTa-v3-small or an equivalent efficient transformer.

Implement:

- configurable training
- checkpointing
- early stopping
- deterministic seed
- class imbalance handling
- gradient accumulation
- validation
- model export

Compare against the TF-IDF baseline.

Never optimize only for accuracy.

For moderation, pay special attention to:

- recall
- precision
- PR-AUC
- false negatives
- false positives

--------------------------------------------------
14. VISION MODEL
--------------------------------------------------

Use a pretrained CLIP or ViT-family encoder.

Start with frozen embeddings.

Only fine-tune if experiments justify it.

Measure:

- classification quality
- embedding generation throughput
- inference latency
- memory usage

--------------------------------------------------
15. MULTIMODAL MODEL
--------------------------------------------------

Implement:

TEXT ENCODER
+
IMAGE ENCODER
↓
PROJECTION
↓
FUSION
↓
CLASSIFICATION HEAD

Start simple.

Recommended first implementation:

text embedding
+
image embedding
↓
concatenation
↓
MLP
↓
policy logits

Do not begin with an unnecessarily complex multimodal transformer.

Run three controlled experiments:

A. Text only
B. Image only
C. Text + image

Compare:

- F1
- macro F1
- PR-AUC
- precision
- recall
- false positives
- false negatives
- inference latency

Save the raw results.

--------------------------------------------------
16. REPRESENTATION LEARNING
--------------------------------------------------

Generate versioned embeddings.

Each embedding must have:

content_id
embedding
embedding_model
embedding_model_version
timestamp

When the model changes, embeddings must be rebuildable.

Never mix embeddings from incompatible model versions without explicitly documenting it.

--------------------------------------------------
17. RETRIEVAL
--------------------------------------------------

Implement:

1. BM25
2. Dense vector retrieval
3. Hybrid retrieval

Dense retrieval:

content
 ↓
embedding
 ↓
FAISS/Qdrant
 ↓
top-K

Lexical:

content
 ↓
BM25
 ↓
top-K

Hybrid:

normalized BM25 score
+
normalized dense score
↓
combined score

The weight must be configurable.

Tune it using validation data rather than choosing it arbitrarily.

--------------------------------------------------
18. RERANKING
--------------------------------------------------

Pipeline:

query
 ↓
BM25 + dense retrieval
 ↓
top 50
 ↓
cross encoder
 ↓
top 10

Evaluate:

- Recall@K
- MRR where applicable
- NDCG where applicable
- reranking latency

Do not claim ranking improvements without measurement.

--------------------------------------------------
19. MULTI-STAGE MODERATION
--------------------------------------------------

Do not automatically run the most expensive model on everything.

Prototype:

Stage 1:
cheap text/image classifier

Stage 2:
multimodal model for uncertain content

Stage 3:
retrieval + reranking for high-risk/uncertain content

Stage 4:
human review

Demonstrate the tradeoff between:

- accuracy
- latency
- compute cost
- review volume

This is an important production design decision.

--------------------------------------------------
20. POLICY / DECISION ENGINE
--------------------------------------------------

Models produce probabilities.

The policy engine produces actions.

Conceptually:

risk score
 ↓
threshold policy
 ↓
ALLOW / REVIEW / BLOCK

Thresholds must be configurable.

Use validation data to study threshold behavior.

Measure:

- false-positive rate
- false-negative rate
- allow rate
- review rate
- block rate

Do not claim thresholds are production optimal.

--------------------------------------------------
21. BUSINESS COST MODEL
--------------------------------------------------

Implement a configurable cost matrix.

Example:

false negative cost
false positive cost
human review cost
latency cost

Use it to compare decision thresholds.

This demonstrates data-driven Trust & Safety reasoning.

Do not invent real-world TikTok costs.

Use clearly labeled hypothetical project costs.

--------------------------------------------------
22. MODEL REGISTRY
--------------------------------------------------

Use MLflow.

Every model must have:

- model version
- Git commit
- dataset version
- training configuration
- evaluation metrics
- artifact
- creation timestamp

Do not use filenames such as:

model_final_v2_really_final.pt

Use proper versioning.

--------------------------------------------------
23. MODEL QUALITY GATES
--------------------------------------------------

Every candidate model must be compared against the current production model.

Example conceptual gate:

F1 >= allowed degradation
Recall >= required minimum
P95 latency <= maximum
No critical policy regression

Thresholds must live in configuration.

If a model fails:

DO NOT DEPLOY.

Save the evaluation report.

--------------------------------------------------
24. REGRESSION TESTING
--------------------------------------------------

Maintain a fixed evaluation set.

Every model candidate must run against:

- general quality set
- policy-specific sets
- difficult examples
- previous failure cases

Track regression by policy.

Example:

HARASSMENT
HATEFUL
VIOLENCE
etc.

A model can improve overall F1 while becoming worse on a critical policy.

The pipeline must detect this.

--------------------------------------------------
25. MODEL CARD
--------------------------------------------------

Every production candidate must have a model card containing:

- purpose
- training data
- evaluation data
- metrics
- known limitations
- failure modes
- intended use
- non-intended use
- model version
- training configuration

Never claim fairness, safety, or robustness without evaluation.

--------------------------------------------------
26. FASTAPI SERVING
--------------------------------------------------

Implement:

POST /moderate
POST /retrieve
POST /feedback

GET /health
GET /ready
GET /metrics
GET /model

Moderation response should contain:

decision
policy
risk_score
model_version
latency
similar_content where applicable

Do not expose sensitive/private information.

--------------------------------------------------
27. TRAINING-SERVING CONSISTENCY
--------------------------------------------------

Use the same preprocessing code for:

- training
- evaluation
- inference

Do not duplicate preprocessing logic.

Shared preprocessing must live in:

src/safelens/data/preprocessing/

Add tests that verify consistent transformations.

--------------------------------------------------
28. MODEL SERVING OPTIMIZATION
--------------------------------------------------

First benchmark PyTorch.

Then, where supported:

PyTorch
 ↓
ONNX
 ↓
ONNX Runtime

Compare:

- P50
- P95
- P99
- throughput
- memory

If ONNX is worse, report the result.

Never manipulate benchmarks.

--------------------------------------------------
29. STORAGE
--------------------------------------------------

Use PostgreSQL for:

- moderation decisions
- model versions
- human feedback
- review results
- audit records

Use Redis for:

- caching
- duplicate-content decisions
- frequently accessed metadata
- optional embedding caching

Use Qdrant only where vector search actually benefits from it.

--------------------------------------------------
30. AUDITABILITY
--------------------------------------------------

Every moderation decision should be traceable.

Store:

content_id
decision
policy
risk_score
model_version
policy_version
timestamp
retrieval_version where applicable

A historical decision must be reproducible as far as practical.

--------------------------------------------------
31. HUMAN FEEDBACK LOOP
--------------------------------------------------

Implement:

POST /feedback

Store:

content_id
model_prediction
model_confidence
human_label
model_version
timestamp

Build hard-example mining.

Prioritize:

- high-confidence incorrect predictions
- low-confidence cases
- policy disagreements
- newly emerging patterns

Generate a retraining dataset from these examples.

Do not automatically retrain without evaluation.

--------------------------------------------------
32. DRIFT MONITORING
--------------------------------------------------

Monitor:

### Data drift

- class distribution
- text length
- image characteristics where practical
- embedding distribution

### Prediction drift

- confidence distribution
- ALLOW/REVIEW/BLOCK rates
- policy distribution

### Model quality

When labels become available:

- precision
- recall
- false-positive rate
- false-negative rate

Separate:

data drift
from
model quality degradation.

--------------------------------------------------
33. ONLINE MONITORING
--------------------------------------------------

Prometheus metrics should include:

moderation_requests_total
moderation_errors_total
moderation_latency_seconds
model_inference_latency_seconds
retrieval_latency_seconds
reranking_latency_seconds
allow_decisions_total
review_decisions_total
block_decisions_total

Also track:

model_version
policy_version
prediction distribution
confidence distribution

--------------------------------------------------
34. GRAFANA
--------------------------------------------------

Create dashboards for:

### Service

- requests/sec
- P50
- P95
- P99
- error rate

### Model

- confidence
- policy distribution
- model version

### Moderation

- ALLOW
- REVIEW
- BLOCK

### Retrieval

- retrieval latency
- reranking latency
- top-K behavior

Every dashboard metric must have a clear operational purpose.

--------------------------------------------------
35. DEPLOYMENT ENVIRONMENTS
--------------------------------------------------

Simulate three environments:

DEV
STAGING
PRODUCTION

Development:

local experimentation.

Staging:

candidate model.

Production:

currently approved model.

Do not allow a candidate model to silently replace production.

--------------------------------------------------
36. SHADOW DEPLOYMENT
--------------------------------------------------

Implement a local shadow simulation.

Production model:
controls actual decision.

Candidate model:
receives the same request.

Candidate output is logged but does not affect the decision.

Compare:

- prediction agreement
- policy differences
- confidence
- latency
- resource usage

--------------------------------------------------
37. CANARY DEPLOYMENT
--------------------------------------------------

Simulate traffic allocation:

production model = 95%
candidate model = 5%

Then:

10%
25%
50%
100%

Only advance if configured health metrics pass.

Otherwise rollback.

Because this is local, simulate traffic routing rather than pretending to have real global traffic.

--------------------------------------------------
38. ROLLBACK
--------------------------------------------------

Rollback must be explicit and tested.

Example:

v1.4 deployed
 ↓
P95 latency regression
 ↓
alert
 ↓
rollback
 ↓
v1.3 becomes active

Document:

- detection
- impact
- root cause
- mitigation
- rollback
- follow-up

--------------------------------------------------
39. INCIDENT PLAYBOOKS
--------------------------------------------------

Create simulated incidents.

At minimum:

### Incident 1
Model quality regression.

### Incident 2
Latency regression.

### Incident 3
Data drift.

### Incident 4
Retrieval failure.

For each incident document:

Detection
Impact
Investigation
Mitigation
Rollback
Root cause
Prevention

--------------------------------------------------
40. LOAD TESTING
--------------------------------------------------

Use Locust.

Test:

10 concurrent users
50 concurrent users
100 concurrent users
200 concurrent users

Measure:

- throughput
- P50
- P95
- P99
- error rate

Run on the M2.

Clearly label results:

"Measured locally on Apple M2."

Never imply TikTok-scale performance.

--------------------------------------------------
41. OPTIONAL STREAMING
--------------------------------------------------

Only after the core system works.

If added:

CONTENT EVENT
 ↓
Kafka
 ↓
Moderation Worker
 ↓
Inference
 ↓
Decision Engine
 ↓
PostgreSQL

Generate synthetic traffic.

Measure consumer throughput and backlog.

Kafka is optional.

Do not sacrifice multimodal ML quality for Kafka.

--------------------------------------------------
42. TESTING
--------------------------------------------------

Unit tests:

- preprocessing
- score normalization
- policy engine
- threshold logic
- hybrid retrieval
- ranking
- request validation

Model tests:

- model loading
- output shape
- probability validity
- regression set

Integration tests:

- FastAPI
- PostgreSQL
- Redis
- Qdrant
- retrieval
- model loading

Serving tests:

- /health
- /ready
- /moderate
- invalid input
- model failure

Tests should run without GPU where possible.

Use mocks for expensive model calls in unit tests.

--------------------------------------------------
43. CI/CD
--------------------------------------------------

GitHub Actions must run:

- formatting
- linting
- type checking
- unit tests
- integration tests where practical

CI must not require a GPU.

A pull request should fail if:

- tests fail
- lint fails
- type checking fails
- critical quality gates fail

--------------------------------------------------
44. GIT WORKFLOW
--------------------------------------------------

Use small, meaningful commits.

Examples:

feat(data): add dataset validation
feat(data): add dataset manifest
feat(model): add text baseline
feat(model): add transformer classifier
feat(model): add vision encoder
feat(model): add multimodal fusion
feat(retrieval): add dense retrieval
feat(retrieval): add BM25 retrieval
feat(ranking): add cross encoder
feat(moderation): add policy engine
feat(api): add moderation endpoint
feat(monitoring): add model metrics
feat(deployment): add shadow evaluation
feat(deployment): add canary simulation
perf(model): benchmark ONNX inference
test(model): add regression suite
docs: add model card

Do not make giant commits.

--------------------------------------------------
45. BENCHMARK HONESTY
--------------------------------------------------

This rule is absolute.

Never fabricate:

- F1
- precision
- recall
- PR-AUC
- latency
- throughput
- dataset size
- cost
- GPU time
- scalability

Every benchmark must have:

- command used
- environment
- configuration
- dataset version
- raw output
- summarized result

Store raw results under:

benchmarks/results/

--------------------------------------------------
46. NEGATIVE RESULTS
--------------------------------------------------

Do not hide failed experiments.

If:

BM25 > dense retrieval

report it.

If:

ONNX > PyTorch

report it.

If:

multimodal < text-only

report it.

If:

parallelism hurts latency

report it.

A documented negative result demonstrates engineering judgment.

--------------------------------------------------
47. DOCUMENTATION
--------------------------------------------------

README must explain:

1. Problem
2. Why the problem matters
3. Architecture
4. Data
5. Policy taxonomy
6. ML models
7. Multimodal architecture
8. Retrieval
9. Ranking
10. Multi-stage moderation
11. Training pipeline
12. Model registry
13. Deployment
14. Shadow deployment
15. Canary deployment
16. Monitoring
17. Feedback loop
18. Drift
19. Benchmarks
20. Limitations
21. Reproducibility

Do not write marketing language.

Write technical documentation.

--------------------------------------------------
48. DEVELOPMENT PHASES
--------------------------------------------------

Do NOT build everything simultaneously.

Follow this order.

PHASE 1
Repository foundation
- Python package
- configuration
- logging
- testing
- CI
- documentation skeleton

PHASE 2
Data
- dataset acquisition
- validation
- preprocessing
- manifests
- leakage checks

PHASE 3
Baseline
- TF-IDF
- Logistic Regression
- evaluation

PHASE 4
Text model
- DeBERTa
- training
- evaluation

PHASE 5
Vision model
- CLIP/ViT
- evaluation

PHASE 6
Multimodal
- fusion
- evaluation
- comparison

PHASE 7
Representation learning
- embeddings
- versioning

PHASE 8
Retrieval
- BM25
- dense
- hybrid

PHASE 9
Ranking
- cross encoder

PHASE 10
Moderation
- risk engine
- thresholds
- policy decisions

PHASE 11
Serving
- FastAPI
- preprocessing
- inference

PHASE 12
Storage
- PostgreSQL
- Redis
- Qdrant

PHASE 13
ML lifecycle
- MLflow
- model registry
- quality gates

PHASE 14
Deployment
- staging
- shadow
- canary
- rollback

PHASE 15
Monitoring
- Prometheus
- Grafana
- model metrics
- data metrics

PHASE 16
Feedback
- human labels
- hard-example mining
- retraining pipeline

PHASE 17
Drift
- data drift
- prediction drift
- quality degradation

PHASE 18
Optimization
- ONNX
- latency
- throughput

PHASE 19
Incident simulation
- regression
- latency
- drift
- retrieval failure

PHASE 20
Optional Kafka/AWS

--------------------------------------------------
49. SCOPE CONTROL
--------------------------------------------------

If time is limited:

STOP after Phase 10 for a strong ML prototype.

STOP after Phase 15 for a strong production-style ML system.

STOP after Phase 18 for a strong portfolio project.

Phase 20 is optional.

Never add infrastructure at the expense of the ML experiments.

--------------------------------------------------
50. CLAUDE CODE BEHAVIOR
--------------------------------------------------

Before every implementation task:

1. Read CLAUDE.md.
2. Inspect the repository.
3. Inspect relevant existing files.
4. Identify dependencies.
5. Identify whether the requested feature already exists.
6. Propose the smallest implementation.
7. Implement it.
8. Add tests.
9. Run tests.
10. Run relevant benchmarks.
11. Update documentation.
12. Report exactly what was done.

Do not silently modify unrelated parts of the repository.

Do not rewrite working code unnecessarily.

Do not introduce dependencies without explaining why they are needed.

Do not create placeholder implementations that look complete.

If a production component is simulated, explicitly label it as a simulation.

--------------------------------------------------
51. FIRST TASK
--------------------------------------------------

When this CLAUDE.md is first loaded:

DO NOT build the entire project.

First:

1. Inspect the repository.
2. Inspect the Python environment.
3. Detect whether the machine is Apple Silicon.
4. Detect MPS availability.
5. Detect CUDA availability.
6. Check Python version.
7. Check installed package managers.
8. Check Git state.
9. Check whether Docker is installed.
10. Check whether Docker Compose is installed.

Then produce:

A. Current repository assessment
B. Environment assessment
C. Proposed implementation plan
D. Proposed dependency strategy
E. Phase 1 file changes

Do not implement Phase 2 or later.

After presenting the plan, implement ONLY Phase 1.

Run:

- tests
- linting if configured
- type checking if configured

Report actual results.

Do not claim success without running the commands.

--------------------------------------------------
52. FINAL PROJECT SUCCESS CRITERIA
--------------------------------------------------

The project is complete only when the following are demonstrable:

DATA
[ ] reproducible dataset pipeline
[ ] data validation
[ ] dataset versioning
[ ] leakage checks
[ ] time-based evaluation

ML
[ ] text baseline
[ ] transformer model
[ ] vision model
[ ] multimodal model
[ ] unimodal vs multimodal experiment
[ ] representation learning

RETRIEVAL
[ ] BM25
[ ] dense retrieval
[ ] hybrid retrieval
[ ] reranking

MODERATION
[ ] policy taxonomy
[ ] risk scoring
[ ] threshold optimization
[ ] multi-stage moderation
[ ] human review path

ML LIFECYCLE
[ ] reproducible training
[ ] model registry
[ ] quality gates
[ ] regression suite
[ ] model card

SERVING
[ ] FastAPI
[ ] input validation
[ ] model versioning
[ ] latency instrumentation
[ ] Docker

DEPLOYMENT
[ ] staging
[ ] shadow deployment
[ ] canary simulation
[ ] rollback

OBSERVABILITY
[ ] service metrics
[ ] model metrics
[ ] data metrics
[ ] drift monitoring
[ ] Grafana dashboard

FEEDBACK
[ ] human feedback
[ ] hard-example mining
[ ] retraining workflow

ENGINEERING
[ ] tests
[ ] CI
[ ] reproducible benchmarks
[ ] incident playbooks
[ ] documentation

--------------------------------------------------
53. FINAL POSITIONING
--------------------------------------------------

The final project should support a truthful engineering statement such as:

"Designed and implemented a production-style multimodal Trust & Safety ML platform covering data validation, transformer-based content classification, multimodal representation learning, hybrid retrieval, reranking, policy-based moderation, online inference, model quality gates, shadow/canary deployment, monitoring, human feedback, and model iteration."

Only use measured metrics where available.

Never claim:
- TikTok-scale
- production deployment at a real company
- real user data
- real TikTok data
- real-world moderation performance

unless those things actually occurred.

The project should demonstrate engineering patterns, not pretend to be a real internal TikTok system.
