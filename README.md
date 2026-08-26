# Movie Review Sentiment Analysis with DistilBERT

---

An AI-powered sentiment analysis tool that classifies movie reviews as positive or negative. This project uses DistilBERT as a base model, fine-tuned on custom data to understand modern slang and contemporary language patterns.

## Features

---

- **Pre-trained DistilBERT Model:** Leverages the power of transformer-based language models
- **Fine-tuned for Modern Language:** Custom training on movie reviews including modern slang and colloquialisms
- **Confidence Scoring:** Provides probability scores for prediction confidence
- **Dual Interface:** Available as both CLI and web-based Streamlit application
- **Comprehensive Evaluation:** Includes accuracy metrics, classification reports, and confusion matrix visualizations

## Project Structure

---

- Module 3 - Load Pretrained model/
- Module 4 - Work with Real Data and Evaluate our Model/
- Module 5 - Interactive Testing an AI Model/
- Module 6 - Build an AI Web Application/
- Module 7 - Discuss Other Considerations for AI Development/
- Module 8 - Create a Visualisation to Evaluate our Model/
  - RunCell1.py ... RunCell11.py (evaluation & confusion matrix steps)
  - confusion_matrix.png (saved output)
- Module 9 - Fine Tuning an AI Model/
  - RunCell1.py ... RunCell9.py (fine-tuning pipeline)
  - sentiment_analyzer_finetuned.py (interactive CLI using the fine-tuned model)
- final-code/ (consolidated Streamlit app)
- requirements.txt (pinned dependency versions)
- README.md (this file)

Each `Module X` folder corresponds to a stage of the course, originally written as Jupyter notebook cells (`RunCell1.py`, `RunCell2.py`, etc.). Run them in numerical order within a folder — later cells in the same module depend on variables created by earlier ones. The `final-code/` folder contains the consolidated, production-facing version of the app.**
---


Each `Module X` folder corresponds to a stage of the course, originally written as Jupyter notebook cells (`RunCell1.py`, `RunCell2.py`, etc.). Run them in numerical order within a folder — later cells in the same module depend on variables created by earlier ones. The `final-code/` folder contains the consolidated, production-facing version of the app.

## Installation

---

1. **Clone the repository**
```bash
git clone https://github.com/mohamednaidarous/Sentimental-Analysis.git
cd Sentimental-Analysis
```

2. **Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

---

### Command-line Interface
```bash
python sentiment_model.py
```

### Streamlit Web Application
```bash
streamlit run sentiment_app.py
```

### Running Module Scripts
Navigate into a module folder and run cells in order:
```bash
cd "Module 8 - Create a Visualisation to Evaluate our Model"
python RunCell1.py
python RunCell2.py
# ...continue in numerical order
```

## Handling Long Reviews

---

DistilBERT has a 512-token limit. For longer reviews, the following strategies are recommended:

**Sliding Window Approach**
- Process the review in overlapping chunks of 512 tokens
- Aggregate predictions from each chunk (voting or averaging confidence scores)
- Provides more comprehensive sentiment analysis for long reviews

**Alternative Model Architectures**

| Model | Benefit |
|---|---|
| Longformer | Supports up to 4,096 tokens |
| BigBird | Efficient attention for longer sequences |
| Hierarchical models | Process paragraph-level sentiments then aggregate |

**Smart Truncation Strategies**
- Keep first and last N tokens (capture introduction and conclusion)
- Attention-based selection of most relevant sentences
- Summary-based preprocessing before classification

## Model Evaluation Results

---

Baseline model (pre-trained, not fine-tuned) evaluated on 1,000 IMDB reviews:

| Metric | Value |
|---|---|
| Accuracy | 88.10% |
| Precision (Negative) | 0.87 |
| Recall (Negative) | 0.90 |
| Precision (Positive) | 0.89 |
| Recall (Positive) | 0.86 |

See `Module 8 - Create a Visualisation to Evaluate our Model/confusion_matrix.png` for the full confusion matrix.

## Troubleshooting

---

**`ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`**
Version mismatch between `transformers` and `huggingface_hub`. Fix with:
```bash
pip install --upgrade --force-reinstall transformers huggingface_hub
```

**`NameError: name 'dataset' is not defined` (or similar for `classifier`, `df`, etc.)**
The `RunCellX.py` files were originally Jupyter notebook cells sharing one session. Running them as separate scripts means each one needs its own setup code (imports, model/dataset loading) repeated at the top.

**`TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'logging_dir'`**
Argument names can shift between `transformers` versions. Remove `logging_dir` from `TrainingArguments` if your installed version doesn't support it — it isn't essential.

**Where do I get `slang_reviews.csv`?**
This custom dataset is used in Module 9 for fine-tuning on slang/informal movie review language. It should be provided as part of the course materials — place it in the `Module 9 - Fine Tuning an AI Model/` folder before running the training scripts.

**Model won't load: `HFValidationError: Repo id must use alphanumeric chars...`**
Check that the `MODEL_PATH` in your script points to the exact folder name your training script saved to (e.g. `./fine_tuned_model`), not a different name.

## Future Improvements

---

- Implement sliding window approach for long reviews
- Add neutral sentiment classification
- Support for multi-language reviews
- Aspect-based sentiment analysis (acting, plot, cinematography)
- API endpoint for integration with other applications
- Mobile-responsive design improvements
- Evaluate alternative models (Longformer, BigBird) for longer text support

## License

---

MIT License — feel free to use, modify, and distribute this project with attribution.

## Model Information

---

| Property | Value |
|---|---|
| Base Model | distilbert-base-uncased-finetuned-sst-2-english |
| Task | Binary Sentiment Classification |
| Labels | POSITIVE / NEGATIVE |
| Max Token Length | 512 |
| Framework | Hugging Face Transformers |

## Contributing

---

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements

---

- [Hugging Face](https://huggingface.co) for the Transformers library and pre-trained models
- [Streamlit](https://streamlit.io) for the web application framework
- [IT Online Learning](https://itonlinelearning.com) for project development and training materials