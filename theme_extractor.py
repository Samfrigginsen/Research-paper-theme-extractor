import pandas as pd
import spacy
import string
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import nltk
from nltk.corpus import stopwords
import os
import sys
import subprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import numpy as np

# Download required NLTK data
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

# Load spaCy model with automatic installation
try:
    nlp = spacy.load("en_core_web_lg")  # Using larger model for better accuracy
except OSError:
    print("Downloading spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")

stopwords = nlp.Defaults.stop_words

# Define 15 specific themes with their key terms
THEMES = {
    0: {
        "name": "Teacher Perspectives, Trust, and Professional Development",
        "keywords": [
            "teacher",
            "professional development",
            "trust",
            "perspective",
            "training",
            "skill",
            "competency",
        ],
    },
    1: {
        "name": "AI Substitution vs. Augmentation of Teacher Roles",
        "keywords": [
            "substitution",
            "augmentation",
            "role",
            "replacement",
            "assistance",
            "support",
            "collaboration",
        ],
    },
    2: {
        "name": "Generative AI Tools (e.g., ChatGPT) in Education",
        "keywords": [
            "chatgpt",
            "generative ai",
            "llm",
            "language model",
            "gpt",
            "openai",
            "text generation",
        ],
    },
    3: {
        "name": "Personalized and Adaptive Learning Systems",
        "keywords": [
            "personalized",
            "adaptive",
            "individual",
            "customized",
            "learning path",
            "student-centered",
        ],
    },
    4: {
        "name": "Ethics, Equity, and Academic Integrity",
        "keywords": [
            "ethics",
            "equity",
            "integrity",
            "fairness",
            "bias",
            "privacy",
            "transparency",
        ],
    },
    5: {
        "name": "AI in Higher Education: Challenges and Opportunities",
        "keywords": [
            "higher education",
            "university",
            "college",
            "challenge",
            "opportunity",
            "institution",
        ],
    },
    6: {
        "name": "Intelligent Tutoring Systems (ITS) and Chatbots",
        "keywords": [
            "intelligent tutoring system",
            "automated tutoring",
            "virtual tutor",
            "chatbot",
            "conversational agent",
            "automated teaching system",
            "adaptive tutoring",
            "computerized tutor",
        ],
    },
    7: {
        "name": "Student Perceptions and Engagement",
        "keywords": [
            "student",
            "perception",
            "engagement",
            "motivation",
            "attitude",
            "experience",
            "feedback",
        ],
    },
    8: {
        "name": "Adoption Barriers and Implementation Strategies",
        "keywords": [
            "barrier",
            "implementation",
            "adoption",
            "strategy",
            "challenge",
            "solution",
            "framework",
        ],
    },
    9: {
        "name": "Future Trends and Speculative Impacts",
        "keywords": [
            "future",
            "trend",
            "impact",
            "speculation",
            "prediction",
            "prospect",
            "development",
        ],
    },
    10: {
        "name": "Human-AI Collaboration and Hybrid Learning",
        "keywords": [
            "collaboration",
            "hybrid",
            "partnership",
            "integration",
            "combined",
            "blended",
            "interaction",
        ],
    },
    11: {
        "name": "Assessment and Learning Analytics",
        "keywords": [
            "assessment",
            "analytics",
            "evaluation",
            "measurement",
            "data",
            "performance",
            "tracking",
        ],
    },
    12: {
        "name": "Curriculum Design and Pedagogical Innovation",
        "keywords": [
            "curriculum",
            "pedagogy",
            "design",
            "innovation",
            "methodology",
            "approach",
            "framework",
        ],
    },
    13: {
        "name": "Regional, Cultural, and Subject-Specific Studies",
        "keywords": [
            "region",
            "culture",
            "subject",
            "specific",
            "context",
            "local",
            "domain",
        ],
    },
    14: {
        "name": "Social and Emotional Implications",
        "keywords": [
            "social",
            "emotional",
            "wellbeing",
            "mental health",
            "interaction",
            "relationship",
            "impact",
        ],
    },
}


def preprocess(text):
    """Preprocess the text with more sophisticated cleaning."""
    if pd.isna(text):
        return ""

    # Convert to lowercase and remove special characters
    text = str(text).lower()
    text = "".join(char for char in text if char.isalnum() or char.isspace())

    # Process with spaCy
    doc = nlp(text)

    # Extract tokens with more sophisticated filtering
    tokens = []
    for token in doc:
        # Skip stopwords, punctuation, and numbers
        if (
            token.text in stopwords
            or token.text in string.punctuation
            or token.is_digit
            or len(token.text) < 2
        ):
            continue

        # Get lemma
        lemma = token.lemma_

        # Skip if lemma is too short or is a number
        if len(lemma) < 2 or lemma.isdigit():
            continue

        tokens.append(lemma)

    return " ".join(tokens)


def get_dominant_topic(lda_model, bow, text):
    """Get the dominant topic with confidence threshold and keyword matching."""
    topics = lda_model.get_document_topics(bow)
    if topics:
        # Sort topics by probability
        sorted_topics = sorted(topics, key=lambda x: x[1], reverse=True)
        text_lower = text.lower()

        for topic_id, prob in sorted_topics:
            if prob > 0.15:
                theme_keywords = THEMES[topic_id]["keywords"]
                matching_keywords = [k for k in theme_keywords if k in text_lower]

                # Require either:
                # - Multiple keyword matches, or
                # - One specific keyword match (longer phrases are more specific)
                if len(matching_keywords) > 1 or any(
                    len(k.split()) > 1 for k in matching_keywords
                ):
                    return topic_id

        # If no strong matches, return highest probability topic
        return sorted_topics[0][0]
    return None


def get_theme_description(text, topic_keywords, dominant_topic):
    """Generate a more detailed theme description."""
    theme_info = THEMES.get(
        dominant_topic, {"name": "General Education", "keywords": []}
    )
    theme = theme_info["name"]

    # Find relevant keywords that appear in the text
    relevant_keywords = [k for k in theme_info["keywords"] if k in text.lower()]

    if relevant_keywords:
        # Get the most relevant keywords (up to 5 instead of 3)
        top_keywords = relevant_keywords[:5]
        theme += f" - Focusing on {', '.join(top_keywords)}"

    return theme


def main():
    print("Reading Excel file...")
    df = pd.read_excel("teacher vs ai papers.xlsx")

    # Combine all relevant text fields
    df["Combined_Text"] = (
        df["Name of Paper"].fillna("")
        + " "
        + df["Key Arguments"].fillna("")
        + " "
        + df["Methodology"].fillna("")
        + " "
        + df["Research Gaps"].fillna("")
        + " "
        + df["Result"].fillna("")
    )

    print("Preprocessing text...")
    df["Processed_Text"] = df["Combined_Text"].apply(preprocess)

    print("Creating dictionary and corpus...")
    texts = [text.split() for text in df["Processed_Text"]]

    # Create bigrams and trigrams with less strict thresholds
    bigram = Phrases(texts, min_count=3, threshold=50)
    trigram = Phrases(bigram[texts], threshold=50)
    texts = [trigram[bigram[text]] for text in texts]

    # Create dictionary with less strict filtering
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=1, no_above=0.7)
    corpus = [dictionary.doc2bow(text) for text in texts]

    print("Building LDA model...")
    num_topics = 15
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=100,  # Increased from 50 to 100 for more thorough training
        random_state=42,
        alpha="auto",  # Automatically learn topic distribution
        eta="auto",  # Automatically learn word distribution
        chunksize=50,  # Decreased from 100 to 50 for more granular processing
        eval_every=None,
        iterations=2000,  # Increased from 1000 to 2000
        gamma_threshold=0.001,
        minimum_probability=0.005,  # Lowered from 0.01 for more topic assignments
    )

    print("Calculating coherence score...")
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
        topn=10,  # Consider more words for coherence calculation
    )
    coherence_score = coherence_model.get_coherence()
    print(f"Coherence Score: {coherence_score}")

    print("Extracting dominant topics...")
    df["Dominant_Topic"] = df.apply(
        lambda row: get_dominant_topic(
            lda_model, corpus[row.name], row["Combined_Text"]
        ),
        axis=1,
    )

    print("Generating theme descriptions...")
    df["Theme"] = df.apply(
        lambda row: get_theme_description(
            row["Combined_Text"], None, row["Dominant_Topic"]
        ),
        axis=1,
    )

    print("Saving results...")
    output_df = df[["Name of Paper", "Theme"]]
    output_df.to_excel(
        "extracted_themes_v2.xlsx", index=False
    )  # Changed filename to indicate version 2
    print("Results saved to extracted_themes_v2.xlsx")

    print("\nTheme Distribution:")
    theme_counts = output_df["Theme"].value_counts()
    for theme, count in theme_counts.items():
        print(f"{theme}: {count} papers")


if __name__ == "__main__":
    main()
