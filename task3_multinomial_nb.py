"""
Task 3: Multinomial Naive Bayes for ArXiv CS subcategory classification.
Uses abstract text only (id is not used as a feature).

Dependencies: pandas, scikit-learn
"""

import json
import os
import re
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB

# ==========================================
# HARDCODED CONFIGURATION
# ==========================================

# main()
RUN_PIPELINE = True                       # Set to False to skip training/prediction
EXPORT_PARAMETERS_ONLY = False            # Set to True to export config and exit
CHECK_TRAINING_SCORE = False              # Set to True to print final train-set score
USE_SAFE_TEXT_CLEANING = True             # Master toggle for safe_clean_arxiv_text()
STRIP_LATEX_MATH = True                   # Sub-toggle: remove inline LaTeX math like $...$
STRIP_URLS = True                         # Sub-toggle: remove raw http/https URLs
NORMALIZE_WHITESPACE = True               # Sub-toggle: collapse whitespace and trim
EXPORT_SUBMISSION = True                 # Set to False to skip Kaggle CSV export
STAMP_PREDICTION_FILE = True              # Append a run_id to the submission filename
LOG_RUN = True                            # Set to False to skip appending to the log file
SKIP_IF_ALREADY_LOGGED = False             # Set to False to allow duplicate runs in the log

DATA_DIR = './'                           # Directory where your CSVs live
OUTPUT_FILE = 'MultinomialNB_Prediction.csv'  # Base name; run_id is inserted before .csv
PARAMETERS_FILE = 'MultinomialNB_Parameters.json'
LOG_FILE = 'MultinomialNB_RunLog.txt'

# train_test_split()
VAL_FRACTION = 0.2                        # Set to 0.0 to skip validation
# RANDOM_STATE = 42
RANDOM_STATE = 69

# TfidfVectorizer()
MAX_FEATURES = 750000                     # Max TF-IDF words
MIN_DF = 2
MAX_DF = 0.85                            # Float: fraction of docs. Use 1.0 to disable.
STOP_WORDS = 'english'                         # 'english' to strip common stopwords, None to keep
NGRAM_RANGE = (1, 3)
SUBLINEAR_TF = False

# Naive Bayes model selection
# 'multinomial' = MultinomialNB (uses ALPHA, FIT_PRIOR; ignores NORM)
# 'complement'  = ComplementNB  (uses ALPHA, FIT_PRIOR, NORM)
MODEL = 'multinomial'
# MODEL = 'complement'
NORM = True                              # CNB only: L2-normalize complement weights
ALPHA = 0.003                           # Laplace smoothing (both models)
FIT_PRIOR = False                         # Learn class priors from the training data

# ==========================================


def _make_classifier():
    """Instantiate the NB estimator chosen by MODEL.

    Centralised so the validation step and the full-fit step stay in sync, and
    so adding new model variants in future is one edit instead of two.
    """
    if MODEL == 'multinomial':
        return MultinomialNB(alpha=ALPHA, fit_prior=FIT_PRIOR)
    if MODEL == 'complement':
        return ComplementNB(alpha=ALPHA, fit_prior=FIT_PRIOR, norm=NORM)
    raise ValueError(f"Unknown MODEL: {MODEL!r}. Use 'multinomial' or 'complement'.")


def get_editable_parameters():
    return {
        'DATA_DIR': DATA_DIR,
        'OUTPUT_FILE': OUTPUT_FILE,
        'PARAMETERS_FILE': PARAMETERS_FILE,
        'RUN_PIPELINE': RUN_PIPELINE,
        'EXPORT_PARAMETERS_ONLY': EXPORT_PARAMETERS_ONLY,
        'CHECK_TRAINING_SCORE': CHECK_TRAINING_SCORE,
        'USE_SAFE_TEXT_CLEANING': USE_SAFE_TEXT_CLEANING,
        'STRIP_LATEX_MATH': STRIP_LATEX_MATH,
        'STRIP_URLS': STRIP_URLS,
        'NORMALIZE_WHITESPACE': NORMALIZE_WHITESPACE,
        'EXPORT_SUBMISSION': EXPORT_SUBMISSION,
        'STAMP_PREDICTION_FILE': STAMP_PREDICTION_FILE,
        'LOG_RUN': LOG_RUN,
        'LOG_FILE': LOG_FILE,
        'MAX_FEATURES': MAX_FEATURES,
        'MODEL': MODEL,
        'NORM': NORM,
        'ALPHA': ALPHA,
        'FIT_PRIOR': FIT_PRIOR,
        'VAL_FRACTION': VAL_FRACTION,
        'RANDOM_STATE': RANDOM_STATE,
        'MIN_DF': MIN_DF,
        'MAX_DF': MAX_DF,
        'STOP_WORDS': STOP_WORDS,
        'NGRAM_RANGE': list(NGRAM_RANGE),
        'SUBLINEAR_TF': SUBLINEAR_TF
    }


def export_editable_parameters():
    with open(PARAMETERS_FILE, 'w', encoding='utf-8') as file:
        json.dump(get_editable_parameters(), file, indent=2)
    print(f'Saved editable parameters to {PARAMETERS_FILE}')


# Keys that define an "experiment" for dedup purposes. Path / control / logging
# flags are deliberately excluded so they can be toggled without faking a new run.
_DEDUP_KEYS = (
    'USE_SAFE_TEXT_CLEANING',
    'STRIP_LATEX_MATH',
    'STRIP_URLS',
    'NORMALIZE_WHITESPACE',
    'VAL_FRACTION',
    'RANDOM_STATE',
    'MAX_FEATURES',
    'MIN_DF',
    'MAX_DF',
    'STOP_WORDS',
    'NGRAM_RANGE',
    'SUBLINEAR_TF',
    'MODEL',
    'NORM',
    'ALPHA',
    'FIT_PRIOR',
)


def _params_signature(params):
    return tuple(str(params.get(key)) for key in _DEDUP_KEYS)


def find_logged_run(params=None):
    """Return the most recent log entry whose parameter signature matches `params`.

    Returns the entry dict (with keys like 'run_id', 'val_macro_f1', etc.) or
    None if no matching entry exists. We return the *last* match so re-runs of
    the same params surface the most recent score.
    """
    if params is None:
        params = get_editable_parameters()
    if not os.path.exists(LOG_FILE):
        return None
    target = _params_signature(params)
    match = None
    with open(LOG_FILE, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_params = entry.get('parameters', {})
            if _params_signature(entry_params) == target:
                match = entry
    return match


def has_been_logged(params=None):
    return find_logged_run(params) is not None


def make_run_id():
    return datetime.now().strftime('%Y%m%dT%H%M%S')


def build_submission_path(run_id):
    """Return the actual submission path for this run.

    If STAMP_PREDICTION_FILE is True, insert `run_id` before the file extension
    so subsequent runs do not overwrite each other's predictions.
    """
    if not STAMP_PREDICTION_FILE:
        return OUTPUT_FILE
    base, ext = os.path.splitext(OUTPUT_FILE)
    return f'{base}_{run_id}{ext}'


def append_run_log(run_id, val_macro_f1, train_accuracy, prediction_file):
    payload = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'val_macro_f1': val_macro_f1,
        'train_accuracy': train_accuracy,
        'prediction_file': prediction_file,
        'parameters': get_editable_parameters()
    }
    with open(LOG_FILE, 'a', encoding='utf-8') as file:
        file.write(json.dumps(payload) + '\n')
    print(f'Appended run summary to {LOG_FILE} (run_id={run_id})')


_LATEX_MATH_RE = re.compile(r'\$.*?\$')
_URL_RE = re.compile(r'http\S+')
_WHITESPACE_RE = re.compile(r'\s+')


def safe_clean_arxiv_text(text):
    if STRIP_LATEX_MATH:
        text = _LATEX_MATH_RE.sub('LATEX_MATH_HERE', text)
    if STRIP_URLS:
        text = _URL_RE.sub('URL_HERE', text)
    if NORMALIZE_WHITESPACE:
        text = _WHITESPACE_RE.sub(' ', text).strip()
    return text


def main():
    if EXPORT_PARAMETERS_ONLY:
        export_editable_parameters()
        return

    if not RUN_PIPELINE:
        print('RUN_PIPELINE is False, skipping training and prediction.')
        return

    if SKIP_IF_ALREADY_LOGGED:
        prior = find_logged_run()
        if prior is not None:
            prior_f1 = prior.get('val_macro_f1')
            prior_acc = prior.get('train_accuracy')
            prior_run_id = prior.get('run_id') or prior.get('timestamp')
            prior_pred = prior.get('prediction_file')

            f1_str = f'{prior_f1:.4f}' if isinstance(prior_f1, (int, float)) else 'n/a'
            acc_str = f'{prior_acc:.4f}' if isinstance(prior_acc, (int, float)) else 'n/a'
            pred_str = prior_pred if prior_pred else 'none'

            print(
                f'Skipping run: matching parameter set already exists in {LOG_FILE}.\n'
                f'  prior run_id        : {prior_run_id}\n'
                f'  prior val_macro_f1  : {f1_str}\n'
                f'  prior train_accuracy: {acc_str}\n'
                f'  prior prediction    : {pred_str}\n'
                f'Set SKIP_IF_ALREADY_LOGGED = False to force a re-run.'
            )
            return

    run_id = make_run_id()
    val_macro_f1 = None
    train_accuracy = None
    prediction_file = None

    # 1. Load Data
    train = pd.read_csv(f'{DATA_DIR}train.csv', sep='\t')
    test = pd.read_csv(f'{DATA_DIR}test.csv', sep='\t')

    X_text = train['abstract'].fillna('').astype(str)
    y = train['label_id'].astype(int)
    X_test_text = test['abstract'].fillna('').astype(str)

    cleaning_active = USE_SAFE_TEXT_CLEANING and (
        STRIP_LATEX_MATH or STRIP_URLS or NORMALIZE_WHITESPACE
    )
    if cleaning_active:
        X_text = X_text.apply(safe_clean_arxiv_text)
        X_test_text = X_test_text.apply(safe_clean_arxiv_text)

    # 2. Optional Validation Step
    if VAL_FRACTION > 0:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_text, y, test_size=VAL_FRACTION, random_state=RANDOM_STATE, stratify=y
        )
        vec_val = TfidfVectorizer(
            max_features=MAX_FEATURES,
            min_df=MIN_DF,
            max_df=MAX_DF,
            stop_words=STOP_WORDS,
            ngram_range=NGRAM_RANGE,
            sublinear_tf=SUBLINEAR_TF
        )
        X_tr_vec = vec_val.fit_transform(X_tr)

        clf_val = _make_classifier()
        clf_val.fit(X_tr_vec, y_tr)

        y_hat = clf_val.predict(vec_val.transform(X_va))
        val_macro_f1 = float(f1_score(y_va, y_hat, average='macro'))
        print(f'Validation macro-F1: {val_macro_f1:.4f}')

    # 3. Full Training & Submission
    needs_full_fit = EXPORT_SUBMISSION or CHECK_TRAINING_SCORE
    if needs_full_fit:
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            min_df=MIN_DF,
            max_df=MAX_DF,
            stop_words=STOP_WORDS,
            ngram_range=NGRAM_RANGE,
            sublinear_tf=SUBLINEAR_TF
        )
        X_full_vec = vectorizer.fit_transform(X_text)

        clf = _make_classifier()
        clf.fit(X_full_vec, y)

        if CHECK_TRAINING_SCORE:
            train_accuracy = float(clf.score(X_full_vec, y))
            print(f'Training accuracy: {train_accuracy:.4f}')

        if EXPORT_SUBMISSION:
            X_test_vec = vectorizer.transform(X_test_text)
            test_preds = clf.predict(X_test_vec)

            submission = pd.DataFrame({
                'id': test['id'].astype(int),
                'label_id': test_preds.astype(int)
            })
            prediction_file = build_submission_path(run_id)
            submission.to_csv(prediction_file, index=False)
            print(f'Saved Kaggle submission to {prediction_file} (run_id={run_id})')
    else:
        print('Skipped full retrain (EXPORT_SUBMISSION and CHECK_TRAINING_SCORE are False).')

    if LOG_RUN:
        append_run_log(run_id, val_macro_f1, train_accuracy, prediction_file)


if __name__ == '__main__':
    main()
