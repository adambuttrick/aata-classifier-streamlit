import csv
import fasttext
import os
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_short, strip_multiple_whitespaces, stem_text, remove_stopwords, strip_numeric, strip_punctuation
from nltk.stem import WordNetLemmatizer

CATEGORY_MODEL_PATH = 'model_category_classification.bin'
CATEGORY_FILE = 'all_category_terms.csv'
SUBJECT_MODEL_PATH = 'model_subject_classification.bin'
SUBJECT_FILE = 'all_index_terms.csv'


def create_dict_from_file(f):
    dict_from_file = {}
    with open(f, errors="ignore") as f_in:
        reader = csv.reader(f_in)
        for row in reader:
            dict_from_file[row[0]] = row[1]
    return dict_from_file


def preprocess_text(text):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,
                      strip_multiple_whitespaces, strip_numeric, remove_stopwords]
    text = preprocess_string(text, CUSTOM_FILTERS)
    lemma = WordNetLemmatizer()
    text = [lemma.lemmatize(t) for t in text]
    text = ' '.join(text)
    return text


def predict_labels(text):
    results = []
    fasttext.FastText.eprint = lambda x: None
    __location__ = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    model = os.path.join(__location__, SUBJECT_MODEL_PATH)
    subj_data = os.path.join(__location__, SUBJECT_FILE)
    subj_dict = create_dict_from_file(subj_data)
    text = preprocess_text(text)
    classifier = fasttext.load_model(model)
    predicted_labels = classifier.predict(text, k=20)
    labels, ratios = predicted_labels[0], predicted_labels[1]
    filtered_labels = [(subj_dict[label.split('*')[1]], ratio)
                       for label, ratio in zip(labels, ratios)]
    sorted_labels = sorted(
        filtered_labels, key=lambda x: x[1], reverse=True)[:10]
    results = [label[0] for label in sorted_labels]
    return results


def predict_categories(text):
    results = []
    fasttext.FastText.eprint = lambda x: None
    __location__ = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    model = os.path.join(__location__, CATEGORY_MODEL_PATH)
    subj_data = os.path.join(__location__, CATEGORY_FILE)
    subj_dict = create_dict_from_file(subj_data)
    text = preprocess_text(text)
    classifier = fasttext.load_model(model)
    predicted_labels = classifier.predict(text, k=2)
    labels, ratios = predicted_labels[0], predicted_labels[1]
    for i, label in enumerate(labels):
        key = subj_dict[label.split('*')[1]]
        results.append(key)
    return results
