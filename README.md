# Sentiment Analysis — final-code

Consolidated, production-facing code for the Movie Review Sentiment Analysis
project. Everything — inference, evaluation, and fine-tuning, in both a web
UI and a command-line tool — lives in a single file: **`app.py`**.

## What's in this folder

| File | Purpose |
|---|---|
| `app.py` | **The main deliverable.** All-in-one script: Streamlit web app (Analyze / Evaluate / Fine-Tune / About tabs) plus four CLI modes. See below. |
| `sentiment_analyzer.py` | Standalone CLI, base model only. Superseded by `app.py --cli-analyze`; kept for reference. |
| `evaluate_model.py` | Standalone evaluation script. Superseded by `app.py --cli-evaluate`; kept for reference. |
| `train_model.py` | Standalone fine-tuning script. Superseded by `app.py --cli-train`; kept for reference. |
| `sentiment_analyzer_finetuned.py` | Standalone CLI, fine-tuned model only. Superseded by `app.py --cli-analyze-finetuned`; kept for reference. |

## Setup

```bash
pip install -r ../requirements.txt
```

(Run from this folder; `requirements.txt` lives in the project root.)

## Usage

### Web app (recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with four tabs:

- **Analyze** — enter a review, get sentiment + confidence. Choose base or
  fine-tuned model (fine-tuned option appears automatically once one exists).
- **Evaluate** — downloads a sample of the IMDB test set, runs the model
  over it, and shows accuracy, a full classification report, and a
  confusion matrix heatmap — all inline in the browser.
- **Fine-Tune** — upload a CSV (`text`, `label` columns) and fine-tune the
  base model on your own data, right from the browser. Saves to
  `./fine_tuned_model` and immediately makes it available in the other tabs.
- **About** — model info and current fine-tuned-model status.

> Evaluate and Fine-Tune do real work (downloading data, running inference
> over hundreds of reviews, training a transformer) and can take anywhere
> from under a minute to several minutes. Keep the browser tab open while
> they run.

### Command line

The same file also works as a CLI, for each individual module's original
behaviour:

```bash
python app.py --cli-analyze                          # base model, interactive
python app.py --cli-analyze-finetuned                 # fine-tuned model, interactive
python app.py --cli-evaluate                          # evaluate on 1000 IMDB reviews, saves confusion_matrix.png
python app.py --cli-evaluate --num-reviews 2000        # evaluate on a custom number of reviews
python app.py --cli-evaluate --use-fine-tuned          # evaluate the fine-tuned model instead
python app.py --cli-train                              # fine-tune on slang_reviews.csv (3 epochs)
python app.py --cli-train --epochs 5                   # fine-tune for a custom number of epochs
```

Running `python app.py` with no flags falls through to the same behaviour
as `streamlit run app.py`.

## Fine-tuning data

To fine-tune (via either the web app's Fine-Tune tab or `--cli-train`), you
need a CSV with two columns:

| text | label |
|---|---|
| "This movie slapped, no cap." | 1 |
| "Total snoozefest, skip it." | 0 |

`label`: `0` = Negative, `1` = Positive. For the CLI, this file must be
named `slang_reviews.csv` and placed in this folder.

## Model paths

- Base model: `distilbert-base-uncased-finetuned-sst-2-english` (downloaded
  from Hugging Face on first use, then cached locally).
- Fine-tuned model: saved to `./fine_tuned_model` after training. This
  folder is excluded from git (see `.gitignore`) since model weights are
  large and reproducible by re-running training.

## Troubleshooting

**Wall of `ModuleNotFoundError: No module named 'torchvision'` when running
`streamlit run app.py`**
Harmless. Streamlit's file-watcher scans every installed `transformers`
submodule (including unrelated vision models) to support auto-reload; many
of those need `torchvision`, which this project doesn't use or require.
The app still runs correctly — check `http://localhost:8501` in your
browser. To silence the noise, run with
`streamlit run app.py --server.fileWatcherType none` (you'll need to
restart manually after editing the file).

**`ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`**
Version mismatch between `transformers` and `huggingface_hub`. Fix with:
```bash
pip install --upgrade --force-reinstall transformers huggingface_hub
```

**`ModuleNotFoundError: No module named 'datasets'` (or `transformers`,
etc.)**
You're not in the right conda environment. Run:
```bash
conda activate sentiment-analysis
```
(Check your prompt shows `(sentiment-analysis)`, not `(base)`.)

**Fine-tuned model won't load: `HFValidationError: Repo id must use
alphanumeric chars...`**
The path doesn't match where training actually saved the model. `app.py`
consistently uses `./fine_tuned_model` everywhere — if you're pointing at
a different folder name, correct it.

**`TypeError: TrainingArguments.__init__() got an unexpected keyword
argument 'logging_dir'`**
Older/newer `transformers` versions vary on this argument. `app.py`
already omits it; if you see this in one of the standalone reference
scripts, remove the `logging_dir` line from `TrainingArguments`.
