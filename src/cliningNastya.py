import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import string

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


def count_sentences(texts: list[str]) -> int:
    sent_count = 0

    for text in texts:
        sentences = sent_tokenize(str(text))
        sent_count += len(sentences)

    return sent_count


def count_words(texts: list[str]) -> int:
    vocabulary = set()

    for text in texts:
        words = word_tokenize(str(text).lower())
        vocabulary.update(words)

    return (vocabulary)


def clean_text(text: str) -> str:
    text = text.lower()
    removal_list = ["summary of clinical history", "chief complaint", "reason for consultation",
                    "history of present illness", "reason for visit", "admission diagnoses"]

    for word_reading in removal_list:
        text = text.replace(word_reading, "")

    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = re.sub('\s+', ' ', text)

    tokens = tokenize_text(text)
    text = ' '.join(tokens)

    return text


def tokenize_text(text: str) -> list[str]:
    tokens = word_tokenize(text)

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    stop_words = set(
        stopwords.words('english') +
        ['abc', 'abcd', 'abcdf', 'xyz', 'abcg', 'subjective', 'history', 'exam', 'year',
         'yesterday', 'week', 'today', 'wife', 'protocol']
    )
    tokens = list(filter(
        lambda x:
        not x.isdigit() and
        x not in stop_words and
        re.fullmatch('[a-z]{3,}', x) and
        len(set(x)) > 1,
        tokens))

    return tokens
