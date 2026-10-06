# Nirnaya-Flash

[Model on Hugging Face](https://huggingface.co/9337aaric/Nirnaya-Flash) ·
[Package on PyPI](https://pypi.org/project/nirnaya/) ·
[Technical write-up](https://biswa-mohapatro.github.io/nirnaya/)

Nirnaya-Flash makes typed decisions about text in a single forward pass. You give it a
state (a string or JSON) and a set of questions; it returns an answer and a calibrated
probability for every allowed outcome. It never generates text, so there is nothing to parse.

| Question type | You provide | You get |
| --- | --- | --- |
| `choice` | 2-10 named options, optionally with descriptions | the chosen option and each option's probability |
| `noul` | a yes/no question, optionally describing each side | the probability that the answer is yes |
| `score` | 2-26 ordered levels, lowest first | the probability-weighted level and each level's probability |

- 421M parameters (ModernBERT-large encoder plus a small decision transformer), English, 512 tokens.
- Every question in a request is answered in one batched forward pass.
- Runs on Apple Silicon with MLX and on CPU, CUDA, or Apple MPS with PyTorch.

## Quickstart

```bash
pip install nirnaya            # Apple Silicon (MLX)
pip install "nirnaya[torch]"   # PyTorch: CPU, CUDA, or Apple MPS
```

```python
from nirnaya.inference.mlx_runtime import MLXFlashRuntime   # or: from nirnaya.hf.runtime import TorchFlashRuntime

model = MLXFlashRuntime.from_pretrained("9337aaric/Nirnaya-Flash")
answers = model.predict(
    {"ticket": "I was charged twice for March. Please refund the duplicate today or I cancel."},
    {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages", "sales": "Pricing"},
        },
        "churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Routine", "Soon", "Immediate"]},
    },
)
print(answers["team"]["decision"], answers["team"]["confidence"])
```

### HTTP server

```bash
nirnaya-serve 9337aaric/Nirnaya-Flash --port 8000
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "I was charged twice. Please fix this ASAP."},
  "questions": {"billing": {"type": "noul", "instructions": "Is this ticket about billing?"}}
}'
```

The server accepts the `POST /v1/systemone` request and response format used by Ollama's
decision API and the TypeSafe SDK, so those clients work by changing their base URL. It binds
to localhost by default; set `NIRNAYA_API_KEY` to require a bearer token. Prompts that would
exceed the 512-token context are rejected rather than silently truncated.

### Transformers

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("9337aaric/Nirnaya-Flash", trust_remote_code=True)
```

The raw model returns option logits; use `nirnaya.hf.runtime.TorchFlashRuntime` for input
formatting, calibration, and typed results.

## Evaluation

All numbers below come from one pre-registered run on held-out test data that was never used
for training, model selection, or calibration. Intervals are 95% case-clustered bootstrap
intervals.

| Benchmark | Accuracy | 95% interval |
| --- | --- | --- |
| Typed Decisions (official test, 2,000 decisions / 400 cases) | 77.1% | 75.0% to 79.2% |
| AG News topic (4 options, 600 rows) | 80.0% | 76.8% to 83.2% |
| DAIR Emotion (6 options, 600 rows) | 53.5% | 49.5% to 57.5% |
| SST-5 sentiment (5-level score, 600 rows) | 32.0% | 28.2% to 35.8% |
| Prompt injections (yes/no, 116 rows) | 98.3% | 95.7% to 100.0% |
| BoolQ (yes/no over a passage, 3,270 rows) | 79.3% | 77.9% to 80.8% |
| Banking77 intents (2-10 candidate options, 21,560 rows) | 96.3% | 95.9% to 96.7% |

**Typed Decisions in detail** (calibrated): choice 74.5%, yes/no 84.2%, score
73.9%; expected calibration error 0.054; Brier score against the reference
probability distributions 0.079. When the model reports confidence of at
least 0.90 it is correct 98.6% of the time (21.7% of
decisions). Shuffling the order of choice options changes the answer on 5.8% of
choice questions.

**Speed** on an Apple M5 Pro (MLX, float16, one process): 1 question: 10.1 ms median; 5 questions: 35.8 ms median; 10 questions: 66.7 ms median; 50 questions: 335.0 ms median

## Training

1. **Foundation:** the ModernBERT-large encoder and a newly initialized decision transformer
   were trained together on typed decisions, intent routing, yes/no reading comprehension,
   preference rating, and Open-Jev decision sets.
2. **Specialization:** four epochs mixing typed decisions with news-topic, emotion, and
   prompt-injection data. News and emotion options were shuffled per example so those answers
   do not depend on option order.
3. **Calibration:** temperatures were fitted on a separate held-out calibration split only.

Training data and licenses:

| Dataset | License |
| --- | --- |
| LocalLLaMA/typed-decisions | Apache-2.0 |
| ZefanCai/Open-Jev (customer-control-v1 excluded) | CC0-1.0 |
| PolyAI/banking77 | CC-BY-4.0 |
| google/boolq | CC-BY-SA-3.0 |
| openbmb/UltraFeedback | MIT |
| google-research-datasets/go_emotions | Apache-2.0 |
| heegyu/news-category-dataset (News Category Dataset, Misra 2022) | CC-BY-4.0 |
| deepset/prompt-injections | Apache-2.0 |

AG News, DAIR Emotion, and SST-5 were used only for evaluation.

## Limitations

- English only, 512 tokens per question (question, options, and state together).
- Choice questions accept at most 10 options; split larger label sets into stages.
- Ordinal `score` questions are the weakest question type; check them on your own data.
- Fine-grained sentiment (SST-5, 32.0%) and emotion in short social-media posts
  (DAIR Emotion, 53.5%) are the weakest benchmarks.
- Reordering choice options changes 5.8% of typed-decision choice answers; keep option
  order stable in production, or average over orderings when it matters.
- Probabilities are calibrated on typed-decision data. On other domains, `confidence` is a
  useful signal but not a guarantee; validate thresholds on your own data before relying on them.
- Not intended as the sole basis for high-stakes decisions about people.

## License

Code and weights are released under Apache-2.0. See `NOTICE` for attribution of the base
model and training datasets. BoolQ is distributed under CC-BY-SA-3.0; the dataset itself is
not included here.
