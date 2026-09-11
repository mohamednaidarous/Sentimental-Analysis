"""
app.py

ALL-IN-ONE sentiment analysis application. This single file contains the
full logic of every module in the project:

    - sentiment_analyzer.py            (base model CLI)
    - evaluate_model.py                (evaluation / confusion matrix)
    - train_model.py                   (fine-tuning)
    - sentiment_analyzer_finetuned.py  (fine-tuned model CLI)
    - app.py                           (Streamlit web UI)

It can be run two different ways:

  1) As a web app (default):
         streamlit run app.py
     Opens a browser UI with Analyze / Evaluate / Fine-Tune / About tabs.

  2) As a command-line tool, for any individual module's behaviour:
         python app.py --cli-analyze              (base model CLI)
         python app.py --cli-analyze-finetuned     (fine-tuned model CLI)
         python app.py --cli-evaluate              (evaluate on IMDB, saves confusion_matrix.png)
         python app.py --cli-train                 (fine-tune on slang_reviews.csv)

Running `python app.py` with no flags launches the same behaviour as
`streamlit run app.py` (Streamlit will show a warning that it isn't being
run via `streamlit run`, but the CLI flags above work either way).
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

BASE_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
FINE_TUNED_PATH = "./fine_tuned_model"
DATA_PATH = "slang_reviews.csv"
MAX_LENGTH = 512
BATCH_SIZE = 32
SEED = 42


# ---------------------------------------------------------------------------
# Shared core logic (used by both the CLI functions and the Streamlit UI)
# ---------------------------------------------------------------------------

def normalise_label(label: str) -> str:
    """Map LABEL_0/LABEL_1 (fine-tuned model) or NEGATIVE/POSITIVE to a
    consistent POSITIVE/NEGATIVE string."""
    return "POSITIVE" if label in ("LABEL_1", "POSITIVE") else "NEGATIVE"


def fine_tuned_model_exists() -> bool:
    return os.path.exists(FINE_TUNED_PATH) and os.path.isfile(
        os.path.join(FINE_TUNED_PATH, "config.json")
    )


def load_base_classifier():
    print("Loading model...")
    classifier = pipeline("sentiment-analysis", model=BASE_MODEL_NAME)
    print("Model loaded successfully!\n")
    return classifier


def load_fine_tuned_classifier(model_path: str = FINE_TUNED_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"'{model_path}' not found. Run --cli-train (or the Fine-Tune "
            "tab) first to produce a fine-tuned model."
        )
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("Fine-tuned model loaded successfully!\n")
    return classifier


def predict_sentiment(classifier, texts, batch_size: int = BATCH_SIZE, progress_cb=None):
    """
    Run the classifier over a list of texts in batches.
    progress_cb(batch_num, total_batches), if given, is called after each batch
    (used to drive a Streamlit progress bar without duplicating this loop).
    """
    predictions = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"Processing {len(texts)} reviews in {total_batches} batches...")

    for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[i:i + batch_size]
        results = classifier(batch, truncation=True, max_length=MAX_LENGTH)
        predictions.extend(results)
        if progress_cb:
            progress_cb(batch_num, total_batches)
        if batch_num % 10 == 0 or batch_num == total_batches:
            print(f"Processed batch {batch_num}/{total_batches}")

    return predictions


def load_eval_dataset(num_reviews: int = 1000, seed: int = SEED):
    print("Loading dataset...")
    dataset = load_dataset("imdb", split="test")
    dataset = dataset.shuffle(seed=seed).select(range(num_reviews))
    print(f"Dataset loaded: {len(dataset)} reviews ready for evaluation!")
    return dataset


def compute_metrics_for_trainer(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


def load_custom_csv(csv_path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"'{csv_path}' not found. Place your slang_reviews.csv "
            "(columns: text, label) in this folder before running training."
        )
    df = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    return df


def run_fine_tuning(df: pd.DataFrame, epochs: int = 3, status_cb=None):
    """
    Core fine-tuning routine, shared by the CLI (--cli-train) and the
    Streamlit Fine-Tune tab. status_cb(str), if given, is called with
    progress messages instead of using print().
    """
    def say(msg):
        print(msg)
        if status_cb:
            status_cb(msg)

    say("Preparing dataset...")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2, seed=SEED)

    say("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)

    say("Tokenizing dataset...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
        )

    tokenized_train = dataset["train"].map(tokenize_function, batched=True)
    tokenized_test = dataset["test"].map(tokenize_function, batched=True)

    say("Configuring training...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics_for_trainer,
    )

    say("Training... this may take a few minutes.")
    trainer.train()

    say("Saving fine-tuned model...")
    trainer.save_model(FINE_TUNED_PATH)
    tokenizer.save_pretrained(FINE_TUNED_PATH)
    say(f"Fine-tuning complete! Model saved to {FINE_TUNED_PATH}")

    return trainer


# ---------------------------------------------------------------------------
# CLI mode 1: base model interactive analysis  (was sentiment_analyzer.py)
# ---------------------------------------------------------------------------

def cli_analyze_loop(classifier, label_prefix=""):
    print("=" * 50)
    print(f"Custom Sentiment Analysis{label_prefix}")
    print("=" * 50)
    print("Enter movie reviews to analyze (or 'quit' to exit)\n")

    while True:
        user_input = input("Enter review: ").strip()

        if user_input.lower() == "quit":
            print("\nThanks for using the sentiment analyzer!")
            break

        if not user_input:
            print("Please enter some text.\n")
            continue

        result = classifier(user_input, truncation=True, max_length=MAX_LENGTH)[0]
        sentiment = normalise_label(result["label"])
        print(f"Sentiment:  {sentiment}")
        print(f"Confidence: {result['score']:.2%}\n")


def cli_analyze():
    """Equivalent of running sentiment_analyzer.py directly."""
    classifier = load_base_classifier()
    cli_analyze_loop(classifier)


# ---------------------------------------------------------------------------
# CLI mode 2: fine-tuned model interactive analysis
# (was sentiment_analyzer_finetuned.py)
# ---------------------------------------------------------------------------

def cli_analyze_finetuned():
    """Equivalent of running sentiment_analyzer_finetuned.py directly."""
    classifier = load_fine_tuned_classifier()
    cli_analyze_loop(classifier, label_prefix=" (Fine-Tuned Model)")


# ---------------------------------------------------------------------------
# CLI mode 3: full evaluation on IMDB  (was evaluate_model.py)
# ---------------------------------------------------------------------------

def cli_evaluate(num_reviews: int = 1000, use_fine_tuned: bool = False):
    """Equivalent of running evaluate_model.py directly."""
    classifier = load_fine_tuned_classifier() if use_fine_tuned else load_base_classifier()
    dataset = load_eval_dataset(num_reviews)

    texts = dataset["text"]
    true_labels = list(dataset["label"])

    print("\nMaking predictions...")
    predictions = predict_sentiment(classifier, texts)
    pred_labels = [1 if normalise_label(p["label"]) == "POSITIVE" else 0 for p in predictions]
    print(f"Converted {len(pred_labels)} predictions to binary labels.\n")

    accuracy = accuracy_score(true_labels, pred_labels)
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Correct Predictions: {sum(np.array(true_labels) == np.array(pred_labels))} / {len(true_labels)}")
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=["Negative", "Positive"]))

    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        cbar_kws={"label": "Number of Reviews"},
        annot_kws={"size": 16, "weight": "bold"},
    )
    plt.xlabel("Predicted Sentiment", fontsize=12, weight="bold")
    plt.ylabel("Actual Sentiment", fontsize=12, weight="bold")
    plt.title("Confusion Matrix - Sentiment Analysis Results\n" + f"Accuracy: {accuracy:.2%}",
               fontsize=14, weight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Confusion matrix saved as 'confusion_matrix.png'!")

    # Confusion matrix breakdown
    tn, fp, fn, tp = cm.ravel()
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX BREAKDOWN")
    print("=" * 70)
    print(f"\nTrue Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")

    # Top misclassifications
    errors = [
        {
            "text": texts[i],
            "true": "Positive" if true == 1 else "Negative",
            "predicted": "Positive" if pred == 1 else "Negative",
            "confidence": predictions[i]["score"],
        }
        for i, (true, pred) in enumerate(zip(true_labels, pred_labels))
        if true != pred
    ]
    high_conf_errors = sorted(
        [e for e in errors if e["confidence"] > 0.9], key=lambda x: x["confidence"], reverse=True
    )[:5]

    print("\n" + "-" * 70)
    print("TOP MISCLASSIFIED (highest confidence mistakes)")
    print("-" * 70)
    for i, error in enumerate(high_conf_errors, 1):
        print(f"\n{i}. {error['text'][:200]}...")
        print(f"    Actual: {error['true']}  |  Predicted: {error['predicted']} ({error['confidence']:.2%})")


# ---------------------------------------------------------------------------
# CLI mode 4: fine-tuning  (was train_model.py)
# ---------------------------------------------------------------------------

def cli_train(epochs: int = 3):
    """Equivalent of running train_model.py directly."""
    df = load_custom_csv()
    print(f"Loaded {len(df)} examples from {DATA_PATH}")
    run_fine_tuning(df, epochs=epochs)


# ---------------------------------------------------------------------------
# Streamlit web UI  (was the Streamlit version of app.py)
# ---------------------------------------------------------------------------

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬", layout="centered")

    @st.cache_resource(show_spinner=False)
    def cached_base_classifier():
        return pipeline("sentiment-analysis", model=BASE_MODEL_NAME)

    def get_fine_tuned_classifier():
        return load_fine_tuned_classifier()

    def render_analyze_tab():
        st.subheader("Analyze a Review")

        model_options = ["Base pre-trained model"]
        if fine_tuned_model_exists():
            model_options.append("Fine-tuned model")

        choice = st.radio("Model to use:", model_options, horizontal=True)

        with st.expander("Example reviews"):
            st.markdown(
                "- *This movie was absolutely brilliant, best film I've seen all year!*\n"
                "- *Total snoozefest, I almost fell asleep halfway through.*\n"
                "- *This movie slapped, no cap.*"
            )

        user_input = st.text_area("Enter a movie review:", height=150, key="analyze_input")

        if st.button("Analyze Sentiment", type="primary"):
            if not user_input.strip():
                st.warning("Please enter a review first.")
                return

            with st.spinner("Loading model and running inference..."):
                classifier = (
                    get_fine_tuned_classifier() if choice == "Fine-tuned model" else cached_base_classifier()
                )
                result = classifier(user_input, truncation=True, max_length=MAX_LENGTH)[0]

            sentiment = normalise_label(result["label"])
            confidence = result["score"]

            if sentiment == "POSITIVE":
                st.success("Sentiment: POSITIVE 😊")
            else:
                st.error("Sentiment: NEGATIVE 😞")

            st.metric("Confidence", f"{confidence:.2%}")
            st.progress(confidence)

    def render_evaluate_tab():
        st.subheader("Evaluate the Model")
        st.caption(
            "Downloads a sample of the IMDB test set and runs the model over "
            "it to compute accuracy, a classification report, and a "
            "confusion matrix."
        )

        model_options = ["Base pre-trained model"]
        if fine_tuned_model_exists():
            model_options.append("Fine-tuned model")

        choice = st.radio("Model to evaluate:", model_options, horizontal=True, key="eval_model_choice")
        num_reviews = st.slider("Number of reviews to evaluate", 100, 2000, 1000, step=100)
        batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=32)

        if st.button("Run Evaluation", type="primary"):
            status_text = st.empty()
            progress_bar = st.progress(0.0)

            with st.spinner("Loading model..."):
                classifier = (
                    get_fine_tuned_classifier() if choice == "Fine-tuned model" else cached_base_classifier()
                )

            with st.spinner("Downloading / loading IMDB dataset..."):
                dataset = load_eval_dataset(num_reviews)

            texts = dataset["text"]
            true_labels = list(dataset["label"])

            def progress_cb(batch_num, total_batches):
                progress_bar.progress(batch_num / total_batches)
                status_text.text(f"Processed batch {batch_num}/{total_batches}")

            predictions = predict_sentiment(classifier, texts, batch_size, progress_cb)
            status_text.text("Done!")

            pred_labels = [1 if normalise_label(p["label"]) == "POSITIVE" else 0 for p in predictions]

            accuracy = accuracy_score(true_labels, pred_labels)
            report = classification_report(
                true_labels, pred_labels, target_names=["Negative", "Positive"], output_dict=True
            )
            cm = confusion_matrix(true_labels, pred_labels)

            st.markdown("### Results")
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{accuracy:.2%}")
            col2.metric("Correct", f"{sum(np.array(true_labels) == np.array(pred_labels))} / {len(true_labels)}")

            st.markdown("**Classification Report**")
            st.dataframe(pd.DataFrame(report).transpose().round(3))

            st.markdown("**Confusion Matrix**")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax,
            )
            ax.set_xlabel("Predicted Sentiment")
            ax.set_ylabel("Actual Sentiment")
            ax.set_title(f"Confusion Matrix (Accuracy: {accuracy:.2%})")
            st.pyplot(fig)

    def render_finetune_tab():
        st.subheader("Fine-Tune on Your Own Data")
        st.caption(
            "Upload a CSV with `text` and `label` columns (label: 0 = "
            "Negative, 1 = Positive) to fine-tune the base model on your "
            "own reviews (e.g. slang-heavy movie reviews)."
        )
        st.warning(
            "Fine-tuning trains a real transformer model and can take "
            "several minutes. Keep this browser tab open while it runs."
        )

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        epochs = st.slider("Epochs", 1, 5, 3)

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            if not {"text", "label"}.issubset(df.columns):
                st.error("CSV must contain 'text' and 'label' columns.")
                return

            st.write(f"Loaded {len(df)} examples.")
            st.dataframe(df.head())

            label_counts = df["label"].value_counts()
            st.write(f"Balance — Negative: {label_counts.get(0, 0)}, Positive: {label_counts.get(1, 0)}")

            if st.button("Start Fine-Tuning", type="primary"):
                status = st.empty()
                with st.spinner("Fine-tuning in progress..."):
                    run_fine_tuning(df, epochs=epochs, status_cb=status.text)

                cached_base_classifier.clear()
                st.success(
                    f"Fine-tuning complete! Model saved to `{FINE_TUNED_PATH}`. "
                    "The Analyze and Evaluate tabs will now offer it as an option."
                )

    def render_about_tab():
        st.subheader("About This Project")
        st.markdown(
            f"""
This app consolidates every stage of the sentiment analysis pipeline into
one interface — and one file, `app.py`.

- **Analyze** — run a single review through the model (base or fine-tuned)
- **Evaluate** — measure accuracy, precision/recall, and confusion matrix
  against a sample of the IMDB test set
- **Fine-Tune** — adapt the base model to your own labelled data, directly
  in the browser

**Base model:** `{BASE_MODEL_NAME}`
**Task:** Binary sentiment classification (POSITIVE / NEGATIVE)
**Max input length:** {MAX_LENGTH} tokens

Fine-tuned model present: {"✅ Yes" if fine_tuned_model_exists() else "❌ No — use the Fine-Tune tab to create one"}

This same file can also be run from the command line for each individual
module's original behaviour:

    python app.py --cli-analyze
    python app.py --cli-analyze-finetuned
    python app.py --cli-evaluate
    python app.py --cli-train
            """
        )

    st.title("🎬 Movie Review Sentiment Analyzer")
    tab1, tab2, tab3, tab4 = st.tabs(["Analyze", "Evaluate", "Fine-Tune", "About"])
    with tab1:
        render_analyze_tab()
    with tab2:
        render_evaluate_tab()
    with tab3:
        render_finetune_tab()
    with tab4:
        render_about_tab()


# ---------------------------------------------------------------------------
# Entry point: dispatch between CLI modes and the Streamlit web app
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis - all modules in one file."
    )
    parser.add_argument("--cli-analyze", action="store_true", help="Interactive CLI using the base model")
    parser.add_argument("--cli-analyze-finetuned", action="store_true", help="Interactive CLI using the fine-tuned model")
    parser.add_argument("--cli-evaluate", action="store_true", help="Evaluate the model against IMDB reviews")
    parser.add_argument("--cli-train", action="store_true", help="Fine-tune the model on slang_reviews.csv")
    parser.add_argument("--num-reviews", type=int, default=1000, help="Number of reviews for --cli-evaluate")
    parser.add_argument("--use-fine-tuned", action="store_true", help="Use the fine-tuned model with --cli-evaluate")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs for --cli-train")

    args, _unknown = parser.parse_known_args()

    if args.cli_analyze:
        cli_analyze()
    elif args.cli_analyze_finetuned:
        cli_analyze_finetuned()
    elif args.cli_evaluate:
        cli_evaluate(num_reviews=args.num_reviews, use_fine_tuned=args.use_fine_tuned)
    elif args.cli_train:
        cli_train(epochs=args.epochs)
    else:
        # No CLI flags given: behave as the Streamlit web app.
        # (This branch also runs when launched via `streamlit run app.py`.)
        run_streamlit_app()


if __name__ == "__main__":
    main()