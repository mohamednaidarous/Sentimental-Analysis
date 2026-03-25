import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Load model (with caching so it only loads once)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english")									
classifier = load_model()

# Title and description
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to analyze its sentiment!")

# Text input
user_input = st.text_area(
    "Your review:",
    placeholder="Type your movie review here...",
    height=150
)

# Analyze button
if st.button("Analyze Sentiment", type="primary"):
    if user_input:
        # Make prediction
        with st.spinner("Analyzing..."):
            result = classifier(user_input)[0]
        
        # Display results
        st.divider()
        
        # Sentiment with color
        if result['label'] == 'POSITIVE':
            st.success(f"**Sentiment:** {result['label']} 😊")
        else:
            st.error(f"**Sentiment:** {result['label']} 😞")

        
        # Confidence
        st.metric("Confidence", f"{result['score']:.2%}")

        
        # Progress bar for confidence
        st.progress(result['score'])

    
    else:
        st.warning("Please enter some text first!")


# Sidebar with examples
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