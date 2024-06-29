# Function to format a single article
import re
import nltk
nltk.download('punkt')  # Ensure the punkt tokenizer models are downloaded
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


def format_article(article):
    # Normalize whitespace and strip leading/trailing whitespace
    article = " ".join(article.split()).strip()

    # Replace unwanted symbols
    article = article.replace("@xcite", "")

    # Regular expression to match @xmath followed by any number of digits
    article = re.sub(r"@xmath\d+", "", article)

    # Regular expression to remove common LaTeX patterns
    # This pattern matches \command[opt]{arg} and \command{arg}, including nested braces
    latex_pattern = r"\\[a-zA-Z]+\*?(?:\[.*?\])?(?:\{.*?\})+"
    article = re.sub(latex_pattern, "", article)

    # Replace double backslashes (which are not LaTeX commands) with a single backslash
    article = article.replace("\\\\", "\\")

    # Remove remaining single backslashes not followed by a letter (assumed not to be LaTeX commands)
    article = re.sub(r"\\(?!\\?[a-zA-Z])", "", article)

    # Segment sentences and tokenize words to handle punctuation and spacing accurately
    sentences = nltk.sent_tokenize(article)
    processed_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        processed_sentence = ' '.join(words)
        processed_sentences.append(processed_sentence)

    # Reassemble the article
    article = ' '.join(processed_sentences)

    # Clean up sequences of commas and periods
    article = re.sub(r"[,]{2,}", ",", article)  # Replace multiple commas with a single one
    article = re.sub(r"[.]{3,}", "...", article)  # Replace multiple periods with ellipsis

    # Clean up spaces before commas and periods
    article = re.sub(r"\s+([,\.])", r"\1", article)

    # Remove extra spaces and correct spacing issues around punctuation
    article = re.sub(r"\s+", " ", article)

    # Prepend "summarize: " and ensure the final article is stripped of leading/trailing spaces
    return "summarize: " + article.strip()