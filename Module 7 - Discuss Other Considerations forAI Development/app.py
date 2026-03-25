import streamlit as st
from transformers import pipeline
import logging
from datetime import datetime

# General Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'streamlit_app_{datetime.now().strftime("%Y-%m-%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security Logging Setup
security_logger = logging.getLogger("security")
security_handler = logging.FileHandler(
    f'security_{datetime.now().strftime("%Y-%m-%d")}.log'
)
security_handler.setFormatter(logging.Formatter(
    '%(asctime)s - SECURITY - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
security_logger.addHandler(security_handler)
security_logger.setLevel(logging.INFO)

def log_security_event(event_type: str, detail: str, level: str = "info"):
    """
    Log a security-relevant event with a timestamp.

    Args:
        event_type: Short category label e.g. 'RATE_LIMIT', 'INVALID_INPUT'
        detail:     Human-readable description of the event
        level:      'info' | 'warning' | 'critical'
    """
    message = f"[{event_type}] {detail}"
    if level == "warning":
        security_logger.warning(message)
    elif level == "critical":
        security_logger.critical(message)
    else:
        security_logger.info(message)

# Page Config 
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Rate Limiting Config 
RATE_LIMIT_MAX    = 10  # Maximum requests allowed per window
RATE_LIMIT_WINDOW = 60  # Window duration in seconds

def init_rate_limit():
    """Initialise rate limit tracking in session state."""
    if "request_count" not in st.session_state:
        st.session_state.request_count = 0
    if "window_start" not in st.session_state:
        st.session_state.window_start = datetime.now()

def check_rate_limit():
    """
    Check whether the current session has exceeded the rate limit.

    Resets the counter automatically when the time window expires.
    Returns (allowed: bool, message: str)
    """
    init_rate_limit()

    now        = datetime.now()
    window_age = (now - st.session_state.window_start).total_seconds()

    # Reset counter if the time window has expired
    if window_age > RATE_LIMIT_WINDOW:
        st.session_state.request_count = 0
        st.session_state.window_start  = now
        logger.info("Rate limit window reset.")

    if st.session_state.request_count >= RATE_LIMIT_MAX:
        seconds_remaining = int(RATE_LIMIT_WINDOW - window_age)
        log_security_event(
            "RATE_LIMIT",
            f"Session exceeded {RATE_LIMIT_MAX} requests. "
            f"{seconds_remaining}s until reset.",
            level="warning"
        )
        return False, (
            f"⏱️ Rate limit reached ({RATE_LIMIT_MAX} requests "
            f"per {RATE_LIMIT_WINDOW} seconds). "
            f"Please wait **{seconds_remaining} seconds** before trying again."
        )

    return True, ""

def increment_request_count():
    """Increment the request counter after a successful analysis."""
    init_rate_limit()
    st.session_state.request_count += 1
    logger.info(f"Request count: {st.session_state.request_count}/{RATE_LIMIT_MAX}")

# Load Model 
@st.cache_resource
def load_model():
    """
    Load the sentiment analysis model with comprehensive error handling.

    Wraps model loading in a try-except block to catch any errors that
    may occur during download or initialisation.
    """
    try:
        logger.info("Attempting to load sentiment analysis model...")
        model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        logger.info("✅ Model loaded successfully.")
        return model

    except OSError as e:
        logger.error(f"OS Error occurred while loading the model: {e}")
        st.error("❌ Failed to load the sentiment analysis model.")
        st.error("**Possible causes:**")
        st.write("• Not enough disk space")
        st.write("• Permission denied")
        st.write("• Corrupted cache files")
        st.write("💡 Try clearing the browser cache and refreshing.")
        st.stop()

    except ConnectionError as e:
        logger.error(f"Connection Error occurred while loading the model: {e}")
        st.error("❌ Cannot connect to download the model.")
        st.error("**Possible causes:**")
        st.write("• No internet connection")
        st.write("• Firewall blocking the connection")
        st.write("• Server temporarily down")
        st.write("💡 Check your internet connection and try again in a few minutes.")
        st.stop()

    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        st.error("❌ An unexpected error occurred while loading the model.")
        st.write("💡 Please try again later or contact support if the issue persists.")
        st.stop()

    return None

# Try to load the model and handle any errors gracefully
try:
    classifier = load_model()
    if classifier is None:
        st.error("Critical: Model failed to initialize.")
        st.stop()
except Exception as e:
    logger.critical(f"Critical initialization error: {e}")
    st.error("🚨 Critical error during app initialization.")
    st.stop()

# Validation Constants 
MAX_LENGTH      = 5000  # Maximum characters allowed for input
MIN_LENGTH      = 10    # Minimum characters required for meaningful analysis
MAX_EMOJI_RATIO = 0.3   # Maximum 30% allowed ratio of emojis to total characters

def validate_input(text):
    """
    Validate user input for sentiment analysis.

    Checks for:
    - Minimum and maximum length
    - Excessive emoji usage
    - Presence of non-text content (URLs, code snippets)
    """
    if not text:
        return False, "⚠️ Please enter some text first!"

    if len(text) < MIN_LENGTH:
        return False, f"⚠️ Please enter a longer review (at least {MIN_LENGTH} characters)."

    if len(text) > MAX_LENGTH:
        return False, (
            f"⚠️ Input exceeds maximum length of {MAX_LENGTH} characters. "
            "Please shorten your review."
        )

    # Check for excessive emojis
    emoji_count = sum(1 for char in text if ord(char) > 127)
    if len(text) > 0 and emoji_count / len(text) > MAX_EMOJI_RATIO:
        return False, "⚠️ Too many emojis detected. Please limit emoji usage for accurate sentiment analysis."

    # Check for non-text content (URLs or code snippets)
    if (
        "http://"       in text
        or "https://"   in text
        or "javascript" in text.lower()
        or "<script"    in text.lower()
    ):
        log_security_event(
            "SUSPICIOUS_INPUT",
            f"Input contained URLs or code-like content. "
            f"First 60 chars: {text[:60]!r}",
            level="warning"
        )
        return False, "❌ Input contains non-text content (like URLs or code). Please enter a valid movie review."

    if len(text) > MAX_LENGTH * 0.9:
        log_security_event(
            "LARGE_INPUT",
            f"Input approaching max length: {len(text)} characters.",
            level="info"
        )

    return True, ""

def sanitize_input(text):
    """
    Sanitize user input to prevent potential security issues.

    Can be expanded to include more comprehensive sanitization such as
    removing HTML tags, escaping special characters, etc.
    """
    import re
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)          # Remove HTML-like tags
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Collapse repeated chars
    text = re.sub(r'\n{3,}', '\n\n', text)       # Limit consecutive line breaks
    return text

# UI 
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to analyze its sentiment!")

user_input = st.text_area(
    "Your review:",
    placeholder="Type your movie review here...",
    height=150
)

# Rate limit status display
init_rate_limit()
requests_left = max(RATE_LIMIT_MAX - st.session_state.request_count, 0)
st.caption(f"🔢 Requests remaining: {requests_left}/{RATE_LIMIT_MAX} (resets every {RATE_LIMIT_WINDOW}s)")

# Analyze button
if st.button("Analyze Sentiment", type="primary"):
    if user_input:

        # Rate limit check
        allowed, rate_message = check_rate_limit()
        if not allowed:
            st.error(rate_message)
            st.stop()

        with st.spinner("Analyzing..."):
            is_valid, message = validate_input(user_input)
            if is_valid:
                result = classifier(sanitize_input(user_input))[0]
                increment_request_count()
                log_security_event(
                    "ANALYSIS_SUCCESS",
                    f"Sentiment analysis completed. "
                    f"Label: {result['label']}, "
                    f"Score: {result['score']:.4f}, "
                    f"Input length: {len(user_input)} chars."
                )
            else:
                log_security_event(
                    "VALIDATION_FAILED",
                    f"Input validation failed: {message} "
                    f"Input length: {len(user_input)} chars.",
                    level="warning"
                )
                st.warning(message)
                st.stop()

        st.divider()

        if result['label'] == 'POSITIVE':
            st.success(f"**Sentiment:** {result['label']} 😊")
        else:
            st.error(f"**Sentiment:** {result['label']} 😞")

        st.metric("Confidence", f"{result['score']:.2%}")
        st.progress(result['score'])

    else:
        st.warning("Please enter some text first!")

#  Sidebar 
st.sidebar.header("Try These Examples:")
st.sidebar.write("Click to copy to clipboard:")

examples = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Terrible waste of time. The plot made no sense.",
    "It was okay, nothing special.",
    "Best film I've seen in years! Highly recommend!",
    "Boring and predictable. Would not watch again."
]

for i, example in enumerate(examples, 1):
    st.sidebar.text_area(f"Example {i}", example, height=80, disabled=True)

# Footer
st.divider()
st.caption("Built with Streamlit and Hugging Face Transformers")