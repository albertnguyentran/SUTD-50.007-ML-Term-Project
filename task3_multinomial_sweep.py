"""
Task 3 sweep harness.

Loads the training data and pre-cleans it once, then iterates over TF-IDF
configurations in the outer loop and Naive Bayes configurations in the inner
loop, writing a row per permutation to a CSV.

Supports two evaluation modes:
- Single split (N_FOLDS = 1): fast, default for wide sweeps.
- Stratified k-fold (N_FOLDS > 1): slower, more stable rankings.

Use `revalidate_top()` to take the best configs from a single-split sweep and
re-evaluate them with k-fold CV for confirmation.

Designed so you can also import these helpers from a Jupyter notebook and reuse
the cached data across many sweeps without reloading.

Dependencies: pandas, scikit-learn
"""

import ast
import csv
import math
import os
import re
import time
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB

# ==========================================
# HARDCODED CONFIGURATION
# ==========================================

# Paths
DATA_DIR = './'
RESULTS_FILE = 'MultinomialNB_Sweep_Results.csv'
CV_RESULTS_FILE = 'MultinomialNB_CV_Results.csv'

# Validation split (used when N_FOLDS == 1)
VAL_FRACTION = 0.2
RANDOM_STATE = 42

# Evaluation mode
# 1 = single stratified train/val split (fast)
# >1 = stratified k-fold CV (more stable, k* slower)
#
# Pinned to 1 so the sweep uses the same split (VAL_FRACTION=0.2,
# RANDOM_STATE=42) as task3_multinomial_nb.py. That makes the val_macro_f1
# column directly comparable to entries in MultinomialNB_RunLog.txt, including
# the current #1 at f1=0.6512, rather than being an independent CV mean.
# AUTO_REVALIDATE (below) will then re-score top hits with k-fold CV.
N_FOLDS = 1

# Cleaning options to sweep over (True = use safe_clean_arxiv_text, False = raw)
# Pinned to True: every top-5 run-log entry used clean=True.
CLEANING_GRID = [True]

# ------------------------------------------------------------------------------
# Targeted neighborhood sweep around the current #1 run-log result:
#     max_features=1_000_000, min_df=2, max_df=0.85, ngram=(1, 2),
#     stop_words='english', sublinear_tf=True, clean=True,
#     model='multinomial', alpha=0.001, fit_prior=False  -> f1=0.6512
#
# Only the three axes most likely to improve on #1 are varied. Everything else
# is pinned to the known-good value. The previous broader grid is preserved
# below (commented) so you can revert.
#
# Permutation count: 3 (max_features) * 3 (max_df) * 3 (alpha) = 27
# ------------------------------------------------------------------------------
TFIDF_GRID = [
    {
        'max_features': max_features,
        'min_df': 2,
        'max_df': max_df,
        'stop_words': 'english',
        'ngram_range': (1, 3),
        'sublinear_tf': False,
    }
    for max_features in (500_000, 650_000, 750_000, 850_000, 1_000_000, 1_500_000)
    for max_df in (0.85,)
]

# Inner loop: Naive Bayes configurations.
#
# Each dict must include 'model' ('multinomial' or 'complement'). MNB respects
# 'fit_prior'; CNB ignores fit_prior and instead respects 'norm'. The harness
# silently drops irrelevant keys before passing kwargs to the estimator.
#
# CNB is disabled for the neighborhood sweep: best CNB run-log entry so far is
# f1~0.52, well below the MNB top. fit_prior=False only, matching #1.
INCLUDE_MULTINOMIAL_NB = True
INCLUDE_COMPLEMENT_NB = False

_MNB_CONFIGS = [
    {'model': 'multinomial', 'alpha': alpha, 'fit_prior': False, 'norm': False}
    for alpha in (0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003)
]

_CNB_CONFIGS = [
    {'model': 'complement', 'alpha': alpha, 'fit_prior': True, 'norm': norm}
    for alpha in (0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.1, 0.3)
    for norm in (False, True)
]

# ------------------------------------------------------------------------------
# Previous broader grid (kept for reference; re-enable by uncommenting these
# blocks and commenting out the neighborhood blocks above):
#
# N_FOLDS = 2
# TFIDF_GRID = [
#     {
#         'max_features': max_features,
#         'min_df': min_df,
#         'max_df': max_df,
#         'stop_words': 'english',
#         'ngram_range': ngram_range,
#         'sublinear_tf': True,
#     }
#     for max_features in (100000, 200000, 500000, 1000000, 2000000)
#     for min_df in (1, 2, 3, 5)
#     for max_df in (0.3, 0.5, 0.85, 0.95)
#     for ngram_range in ((1, 1), (1, 2))
# ]
# INCLUDE_COMPLEMENT_NB = True
# _MNB_CONFIGS = [
#     {'model': 'multinomial', 'alpha': alpha, 'fit_prior': False, 'norm': False}
#     for alpha in (0.0001, 0.001, 0.005, 0.01, 0.03)
# ]
# ------------------------------------------------------------------------------

NB_GRID = (
    (_MNB_CONFIGS if INCLUDE_MULTINOMIAL_NB else [])
    + (_CNB_CONFIGS if INCLUDE_COMPLEMENT_NB else [])
)

# Re-validation defaults
# Threshold tuned around the current #1 (single-split f1=0.6512). Anything
# that beats or matches that on the single split gets promoted to k-fold CV.
REVALIDATE_TOP_N = 3
REVALIDATE_N_FOLDS = 3
REVALIDATE_THRESHOLD = 0.65               # Promotion bar: single-split f1 must >= this
CV_PASS_THRESHOLD = 0.65                  # CV pass bar: k-fold mean f1 must >= this to count
                                          # as PASS; below it is marked REGRESSED. A small
                                          # margin below the promotion bar accounts for CV
                                          # typically scoring slightly lower than the tuned
                                          # single split.
AUTO_REVALIDATE = True                    # After a single-split sweep, auto re-validate
                                          # configs scoring >= REVALIDATE_THRESHOLD with CV

# ==========================================

# Precompiled regexes for the cleaner
_LATEX_MATH_RE = re.compile(r'\$.*?\$')
_URL_RE = re.compile(r'http\S+')
_WHITESPACE_RE = re.compile(r'\s+')


def clean_text(text):
    text = _LATEX_MATH_RE.sub(' ', text)
    text = _URL_RE.sub(' ', text)
    text = _WHITESPACE_RE.sub(' ', text).strip()
    return text


def load_data(data_dir=DATA_DIR, val_fraction=VAL_FRACTION, random_state=RANDOM_STATE):
    """
    Load train.csv once, build cleaned and raw versions, and stratify-split
    each into a fixed (train, val) using the same seed.

    Returns a dict keyed by `use_clean` -> {'text': Series, 'y': Series,
    'split': (X_tr, X_va, y_tr, y_va)}.
    """
    print('Loading train.csv ...')
    train = pd.read_csv(
        f'{data_dir}train.csv',
        sep='\t',
        usecols=['abstract', 'label_id'],
    )

    raw_text = train['abstract'].fillna('').astype(str)
    y = train['label_id'].astype(int)

    print('Cleaning text once ...')
    cleaned_text = raw_text.apply(clean_text)

    text_versions = {False: raw_text, True: cleaned_text}

    data = {}
    for use_clean, text in text_versions.items():
        X_tr, X_va, y_tr, y_va = train_test_split(
            text,
            y,
            test_size=val_fraction,
            random_state=random_state,
            stratify=y,
        )
        data[use_clean] = {
            'text': text,
            'y': y,
            'split': (X_tr, X_va, y_tr, y_va),
        }
    return data


# Fields that uniquely identify an evaluated permutation in the CSV. `model` and
# `norm` were added when ComplementNB was introduced; old rows (which were all
# MultinomialNB) are migrated to model='multinomial', norm='False' by
# `_ensure_schema` so they keep matching their original signatures.
_SIGNATURE_FIELDS = (
    'use_safe_text_cleaning',
    'max_features', 'min_df', 'max_df', 'stop_words', 'ngram_range', 'sublinear_tf',
    'model', 'alpha', 'fit_prior', 'norm',
    'n_folds',
)

_CSV_FIELDNAMES = [
    'timestamp',
    'source',                       # 'sweep' or 'revalidation'
    'use_safe_text_cleaning',
    'max_features', 'min_df', 'max_df', 'stop_words', 'ngram_range', 'sublinear_tf',
    'model',                        # 'multinomial' or 'complement'
    'alpha', 'fit_prior', 'norm',   # fit_prior is for MNB only, norm is for CNB only
    'n_folds',
    'val_macro_f1', 'val_macro_f1_std', 'fit_seconds',
    # Annotation fields. Populated for revalidation rows directly, and back-filled
    # onto matching sweep rows so the sweep CSV becomes a single highlight view.
    'original_val_macro_f1',        # the single-split score that triggered promotion
    'delta_vs_original',            # cv_mean - original (positive = CV agrees / improved)
    'cv_status',                    # '', 'PASS', or 'REGRESSED'
    'cv_macro_f1',                  # only set on sweep rows after annotation
]


def _nb_field(nb_kwargs, key, default):
    """Read a NB hyperparameter, falling back to a default if absent.

    Keeps `_signature` and `_row_for_writer` simple while letting individual NB
    dicts omit keys that don't apply to their model (e.g. CNB rows can omit
    `fit_prior`, MNB rows can omit `norm`).
    """
    value = nb_kwargs.get(key, default)
    return default if value is None else value


def _signature(use_clean, tfidf_kwargs, nb_kwargs, n_folds):
    """Build a hashable signature that uniquely identifies a permutation."""
    return (
        str(use_clean),
        str(tfidf_kwargs['max_features']),
        str(tfidf_kwargs['min_df']),
        str(tfidf_kwargs['max_df']),
        str(tfidf_kwargs['stop_words']),
        str(tfidf_kwargs['ngram_range']),
        str(tfidf_kwargs['sublinear_tf']),
        str(_nb_field(nb_kwargs, 'model', 'multinomial')),
        str(nb_kwargs['alpha']),
        str(_nb_field(nb_kwargs, 'fit_prior', True)),
        str(_nb_field(nb_kwargs, 'norm', False)),
        str(n_folds),
    )


def _load_completed_signatures(results_file):
    """Read the existing results CSV (if any) and return the set of signatures
    that have already been evaluated."""
    if not os.path.exists(results_file):
        return set()
    completed = set()
    with open(results_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                completed.add(tuple(str(row.get(field, '')) for field in _SIGNATURE_FIELDS))
            except KeyError:
                continue
    return completed


def _ensure_schema(path):
    """Migrate an existing CSV to the current `_CSV_FIELDNAMES`.

    If the file's header is outdated (added columns since it was first written),
    re-write it once with the new header so subsequent appends stay aligned.
    Existing rows have missing columns back-filled with empty strings; nothing
    is dropped or modified beyond shape.
    """
    if not os.path.exists(path):
        return
    with open(path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        existing = list(reader.fieldnames or [])
        if existing == _CSV_FIELDNAMES:
            return
        rows = list(reader)

    # Defaults applied to old rows for newly added columns. Anything not listed
    # here is back-filled with empty string. These specific defaults preserve
    # signature identity for legacy rows: every previously-recorded run was a
    # MultinomialNB with fit_prior controlled per-row, so model and norm get
    # the values that match what those rows would have produced under the new
    # signature scheme.
    legacy_defaults = {
        'model': 'multinomial',
        'norm': 'False',
    }

    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            new_row = {}
            for field in _CSV_FIELDNAMES:
                if field in row and row[field] != '':
                    new_row[field] = row[field]
                else:
                    new_row[field] = legacy_defaults.get(field, '')
            writer.writerow(new_row)
    added = [f for f in _CSV_FIELDNAMES if f not in existing]
    print(f'Migrated {path} schema (added columns: {added or "none"}).')


def _make_classifier(nb_kwargs):
    """Dispatch on `nb_kwargs['model']` and return a fresh estimator instance.

    Only the keys that the chosen estimator actually accepts are passed through.
    This lets the NB grid carry a uniform shape (every dict has model/alpha/
    fit_prior/norm) without sklearn complaining about unknown kwargs.
    """
    model = _nb_field(nb_kwargs, 'model', 'multinomial')
    if model == 'multinomial':
        return MultinomialNB(
            alpha=float(nb_kwargs['alpha']),
            fit_prior=bool(_nb_field(nb_kwargs, 'fit_prior', True)),
        )
    if model == 'complement':
        return ComplementNB(
            alpha=float(nb_kwargs['alpha']),
            fit_prior=bool(_nb_field(nb_kwargs, 'fit_prior', True)),
            norm=bool(_nb_field(nb_kwargs, 'norm', False)),
        )
    raise ValueError(f"Unknown NB model: {model!r}. Use 'multinomial' or 'complement'.")


def _row_for_writer(
    use_clean, tfidf_kwargs, nb_kwargs, n_folds,
    val_macro_f1, val_macro_f1_std, fit_seconds,
    source='sweep', original_val_macro_f1=None, cv_status='', cv_macro_f1=None,
):
    if original_val_macro_f1 is not None:
        delta = round(val_macro_f1 - float(original_val_macro_f1), 4)
    else:
        delta = ''
    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'source': source,
        'use_safe_text_cleaning': use_clean,
        'max_features': tfidf_kwargs['max_features'],
        'min_df': tfidf_kwargs['min_df'],
        'max_df': tfidf_kwargs['max_df'],
        'stop_words': tfidf_kwargs['stop_words'],
        'ngram_range': str(tfidf_kwargs['ngram_range']),
        'sublinear_tf': tfidf_kwargs['sublinear_tf'],
        'model': _nb_field(nb_kwargs, 'model', 'multinomial'),
        'alpha': nb_kwargs['alpha'],
        'fit_prior': _nb_field(nb_kwargs, 'fit_prior', True),
        'norm': _nb_field(nb_kwargs, 'norm', False),
        'n_folds': n_folds,
        'val_macro_f1': val_macro_f1,
        'val_macro_f1_std': '' if val_macro_f1_std is None else val_macro_f1_std,
        'fit_seconds': round(fit_seconds, 3),
        'original_val_macro_f1': '' if original_val_macro_f1 is None else original_val_macro_f1,
        'delta_vs_original': delta,
        'cv_status': cv_status,
        'cv_macro_f1': '' if cv_macro_f1 is None else cv_macro_f1,
    }


def _evaluate_single_split(data_entry, tfidf_kwargs, nb_grid, completed, use_clean, results_writer, results_file_handle, log_prefix):
    """Run all NB configs against one (X_tr, X_va, y_tr, y_va) split."""
    X_tr, X_va, y_tr, y_va = data_entry['split']

    pending = [
        nb_kwargs
        for nb_kwargs in nb_grid
        if _signature(use_clean, tfidf_kwargs, nb_kwargs, 1) not in completed
    ]
    if not pending:
        print(f'{log_prefix} skip vectorizer (all NB permutations already done)')
        return 0, len(nb_grid)

    vec = TfidfVectorizer(**tfidf_kwargs)
    Xtr_vec = vec.fit_transform(X_tr)
    Xva_vec = vec.transform(X_va)

    written = 0
    skipped = 0
    for nb_kwargs in nb_grid:
        sig = _signature(use_clean, tfidf_kwargs, nb_kwargs, 1)
        if sig in completed:
            skipped += 1
            continue

        start = time.perf_counter()
        clf = _make_classifier(nb_kwargs).fit(Xtr_vec, y_tr)
        y_hat = clf.predict(Xva_vec)
        elapsed = time.perf_counter() - start

        val_f1 = float(f1_score(y_va, y_hat, average='macro'))
        row = _row_for_writer(
            use_clean, tfidf_kwargs, nb_kwargs, 1,
            val_f1, None, elapsed,
        )
        results_writer.writerow(row)
        results_file_handle.flush()
        completed.add(sig)
        written += 1

        print(
            f'{log_prefix} f1={val_f1:.4f} '
            f'model={_nb_field(nb_kwargs, "model", "multinomial")} '
            f'alpha={nb_kwargs["alpha"]} '
            f'fit_prior={_nb_field(nb_kwargs, "fit_prior", True)} '
            f'norm={_nb_field(nb_kwargs, "norm", False)}'
        )
    return written, skipped


def _evaluate_kfold(
    data_entry, tfidf_kwargs, nb_grid, completed, use_clean, n_folds, random_state,
    results_writer, results_file_handle, log_prefix,
    source='sweep', original_score=None, cv_pass_threshold=None,
):
    """Run all NB configs against `n_folds` stratified folds, fitting TF-IDF
    once per fold and reusing the cached matrices for every NB config.

    `source`, `original_score`, and `cv_pass_threshold` are only meaningful when
    this is invoked from `revalidate_top()`; they let us tag each CV row with
    the single-split score that triggered it and a PASS/REGRESSED status.
    """
    text = data_entry['text']
    y = data_entry['y']

    pending = [
        nb_kwargs
        for nb_kwargs in nb_grid
        if _signature(use_clean, tfidf_kwargs, nb_kwargs, n_folds) not in completed
    ]
    if not pending:
        print(f'{log_prefix} skip vectorizer (all NB permutations already done)')
        return 0, len(nb_grid)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_caches = []
    for tr_idx, va_idx in skf.split(text, y):
        X_tr_text = text.iloc[tr_idx]
        X_va_text = text.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]
        vec = TfidfVectorizer(**tfidf_kwargs)
        Xtr_vec = vec.fit_transform(X_tr_text)
        Xva_vec = vec.transform(X_va_text)
        fold_caches.append((Xtr_vec, y_tr, Xva_vec, y_va))

    written = 0
    skipped = 0
    for nb_kwargs in nb_grid:
        sig = _signature(use_clean, tfidf_kwargs, nb_kwargs, n_folds)
        if sig in completed:
            skipped += 1
            continue

        start = time.perf_counter()
        fold_f1s = []
        for Xtr_vec, y_tr, Xva_vec, y_va in fold_caches:
            clf = _make_classifier(nb_kwargs).fit(Xtr_vec, y_tr)
            y_hat = clf.predict(Xva_vec)
            fold_f1s.append(float(f1_score(y_va, y_hat, average='macro')))
        elapsed = time.perf_counter() - start

        mean_f1 = sum(fold_f1s) / len(fold_f1s)
        var = sum((s - mean_f1) ** 2 for s in fold_f1s) / len(fold_f1s)
        std_f1 = math.sqrt(var)

        cv_status = ''
        if cv_pass_threshold is not None:
            cv_status = 'PASS' if mean_f1 >= cv_pass_threshold else 'REGRESSED'

        row = _row_for_writer(
            use_clean, tfidf_kwargs, nb_kwargs, n_folds,
            mean_f1, round(std_f1, 4), elapsed,
            source=source,
            original_val_macro_f1=original_score,
            cv_status=cv_status,
        )
        results_writer.writerow(row)
        results_file_handle.flush()
        completed.add(sig)
        written += 1

        status_tag = f' [{cv_status}]' if cv_status else ''
        print(
            f'{log_prefix} f1={mean_f1:.4f} (+/- {std_f1:.4f}){status_tag} '
            f'model={_nb_field(nb_kwargs, "model", "multinomial")} '
            f'alpha={nb_kwargs["alpha"]} '
            f'fit_prior={_nb_field(nb_kwargs, "fit_prior", True)} '
            f'norm={_nb_field(nb_kwargs, "norm", False)}'
        )
    return written, skipped


def run_sweep(
    data,
    cleaning_grid=None,
    tfidf_grid=None,
    nb_grid=None,
    results_file=RESULTS_FILE,
    n_folds=N_FOLDS,
    random_state=RANDOM_STATE,
):
    """
    Run a full sweep against the supplied `data` dict (from `load_data()`).

    Single-split mode (n_folds == 1): one TF-IDF fit per outer config, cheap NB
    fits inside.

    K-fold mode (n_folds >= 2): k TF-IDF fits per outer config, then NB grid
    runs against all k cached matrices. Logs mean and std of macro-F1.

    Resumable: on restart, any permutation whose signature is already present
    in `results_file` is skipped. Each row is flushed on write so partial runs
    are preserved if you cancel.
    """
    cleaning_grid = CLEANING_GRID if cleaning_grid is None else cleaning_grid
    tfidf_grid = TFIDF_GRID if tfidf_grid is None else tfidf_grid
    nb_grid = NB_GRID if nb_grid is None else nb_grid

    if n_folds < 1:
        raise ValueError(f'n_folds must be >= 1 (got {n_folds})')

    _ensure_schema(results_file)
    write_header = not os.path.exists(results_file)
    completed = _load_completed_signatures(results_file)

    total = len(cleaning_grid) * len(tfidf_grid) * len(nb_grid)
    already_done = sum(
        1
        for use_clean in cleaning_grid
        for tfidf_kwargs in tfidf_grid
        for nb_kwargs in nb_grid
        if _signature(use_clean, tfidf_kwargs, nb_kwargs, n_folds) in completed
    )
    remaining = total - already_done
    print(
        f'Sweep size: {total} permutations, n_folds={n_folds} '
        f'({already_done} already done, {remaining} to run)'
    )
    if remaining == 0:
        print('Nothing to do. Delete or move the results file to start over.')
        return

    counter = 0
    skipped = 0
    with open(results_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for use_clean in cleaning_grid:
            for tfidf_kwargs in tfidf_grid:
                log_prefix = (
                    f'[clean={use_clean} mf={tfidf_kwargs["max_features"]} '
                    f'mindf={tfidf_kwargs["min_df"]} '
                    f'ngram={tfidf_kwargs["ngram_range"]}]'
                )
                if n_folds == 1:
                    written, skip = _evaluate_single_split(
                        data[use_clean], tfidf_kwargs, nb_grid, completed,
                        use_clean, writer, file, log_prefix,
                    )
                else:
                    written, skip = _evaluate_kfold(
                        data[use_clean], tfidf_kwargs, nb_grid, completed,
                        use_clean, n_folds, random_state,
                        writer, file, log_prefix,
                    )
                counter += written
                skipped += skip

    print(
        f'Done. Wrote {counter} new rows to {results_file} '
        f'(skipped {skipped} already-done permutations).'
    )

    if n_folds == 1 and AUTO_REVALIDATE:
        print(
            f'AUTO_REVALIDATE on: promoting configs with val_macro_f1 >= '
            f'{REVALIDATE_THRESHOLD} to {REVALIDATE_N_FOLDS}-fold CV ...'
        )
        revalidate_top(
            data,
            top_n=None,
            n_folds=REVALIDATE_N_FOLDS,
            min_score=REVALIDATE_THRESHOLD,
            source_results_file=results_file,
            random_state=random_state,
            cv_pass_threshold=CV_PASS_THRESHOLD,
        )


def _row_to_kwargs(row):
    """Parse a CSV row from RESULTS_FILE back into (use_clean, tfidf_kwargs, nb_kwargs)."""
    use_clean = str(row['use_safe_text_cleaning']).lower() in ('true', '1')
    stop_words_raw = row['stop_words']
    stop_words = None if str(stop_words_raw).lower() in ('', 'none', 'nan') else str(stop_words_raw)
    tfidf_kwargs = {
        'max_features': int(row['max_features']),
        'min_df': int(row['min_df']),
        'max_df': float(row['max_df']),
        'stop_words': stop_words,
        'ngram_range': tuple(ast.literal_eval(row['ngram_range'])),
        'sublinear_tf': str(row['sublinear_tf']).lower() in ('true', '1'),
    }
    model_raw = str(row.get('model', '') or 'multinomial').lower()
    model = model_raw if model_raw in ('multinomial', 'complement') else 'multinomial'
    nb_kwargs = {
        'model': model,
        'alpha': float(row['alpha']),
        'fit_prior': str(row.get('fit_prior', 'True')).lower() in ('true', '1'),
        'norm': str(row.get('norm', 'False')).lower() in ('true', '1'),
    }
    return use_clean, tfidf_kwargs, nb_kwargs


def revalidate_top(
    data,
    top_n=REVALIDATE_TOP_N,
    n_folds=REVALIDATE_N_FOLDS,
    min_score=None,
    source_results_file=RESULTS_FILE,
    cv_results_file=CV_RESULTS_FILE,
    sort_by='val_macro_f1',
    random_state=RANDOM_STATE,
    cv_pass_threshold=CV_PASS_THRESHOLD,
):
    """
    Read `source_results_file` and re-evaluate selected configs with `n_folds`
    stratified CV. Selection rules:

    - If `min_score` is set, only consider rows with `sort_by >= min_score`.
    - If `top_n` is set (and not None), then take at most that many of the
      remaining rows, sorted by `sort_by` descending.
    - Setting `top_n=None` and `min_score=0.65` revalidates *everything* above
      the threshold.

    Appends to `cv_results_file`. Resumable: configs already present there
    (matched on the full signature including `n_folds`) are skipped.
    """
    if not os.path.exists(source_results_file):
        print(f'{source_results_file} does not exist; nothing to revalidate.')
        return

    df = pd.read_csv(source_results_file)
    if df.empty:
        print(f'{source_results_file} is empty; nothing to revalidate.')
        return

    if min_score is not None:
        df = df[df[sort_by] >= min_score]
        if df.empty:
            print(f'No rows in {source_results_file} have {sort_by} >= {min_score}.')
            return

    df_sorted = df.sort_values(sort_by, ascending=False)
    if top_n is not None:
        df_sorted = df_sorted.head(top_n)

    print(
        f'Re-validating {len(df_sorted)} configs from {source_results_file} '
        f'with {n_folds}-fold CV '
        f'(min_score={min_score}, top_n={top_n}) ...'
    )

    _ensure_schema(cv_results_file)
    write_header = not os.path.exists(cv_results_file)
    completed = _load_completed_signatures(cv_results_file)

    written = 0
    skipped = 0
    with open(cv_results_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for _, row in df_sorted.iterrows():
            use_clean, tfidf_kwargs, nb_kwargs = _row_to_kwargs(row)
            sig = _signature(use_clean, tfidf_kwargs, nb_kwargs, n_folds)
            if sig in completed:
                skipped += 1
                continue

            log_prefix = (
                f'[revalidate clean={use_clean} mf={tfidf_kwargs["max_features"]} '
                f'mindf={tfidf_kwargs["min_df"]} ngram={tfidf_kwargs["ngram_range"]}]'
            )
            sub_written, _ = _evaluate_kfold(
                data[use_clean], tfidf_kwargs, [nb_kwargs], completed,
                use_clean, n_folds, random_state,
                writer, file, log_prefix,
                source='revalidation',
                original_score=float(row[sort_by]),
                cv_pass_threshold=cv_pass_threshold,
            )
            written += sub_written

    print(
        f'Re-validation done. Wrote {written} new rows to {cv_results_file} '
        f'(skipped {skipped} already-validated configs).'
    )

    annotated = _annotate_sweep_with_revalidation_status(
        sweep_file=source_results_file,
        cv_file=cv_results_file,
    )
    _print_revalidation_summary(cv_results_file, cv_pass_threshold, annotated)


# Subset of signature fields used to match a sweep row to a CV row regardless
# of `n_folds` (because the sweep was n_folds=1 and the CV was n_folds=k).
_PARAM_ONLY_FIELDS = (
    'use_safe_text_cleaning',
    'max_features', 'min_df', 'max_df', 'stop_words', 'ngram_range', 'sublinear_tf',
    'model', 'alpha', 'fit_prior', 'norm',
)


def _param_key(row):
    """Build a `_PARAM_ONLY_FIELDS` tuple for matching across sweep and CV files."""
    return tuple(str(row.get(field, '')) for field in _PARAM_ONLY_FIELDS)


def _annotate_sweep_with_revalidation_status(sweep_file=RESULTS_FILE, cv_file=CV_RESULTS_FILE):
    """Back-fill `cv_macro_f1` and `cv_status` on sweep rows that have a
    matching CV result.

    Workflow:
    1. Load the CV file and build {param_key -> (best_cv_mean, best_cv_status)}.
       If the same params got revalidated more than once (e.g. different fold
       counts), keep the highest mean — that is the most generous read.
    2. Load the sweep file, fill the two columns for any matching row.
    3. Rewrite the sweep file with the full schema.

    Returns the number of sweep rows that were annotated.
    """
    if not os.path.exists(sweep_file) or not os.path.exists(cv_file):
        return 0

    cv_map = {}
    with open(cv_file, 'r', newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            try:
                cv_mean = float(row.get('val_macro_f1', ''))
            except (TypeError, ValueError):
                continue
            key = _param_key(row)
            cv_status = row.get('cv_status', '') or ''
            best = cv_map.get(key)
            if best is None or cv_mean > best[0]:
                cv_map[key] = (cv_mean, cv_status)

    if not cv_map:
        return 0

    with open(sweep_file, 'r', newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))

    annotated = 0
    for row in rows:
        key = _param_key(row)
        if key in cv_map:
            cv_mean, cv_status = cv_map[key]
            row['cv_macro_f1'] = cv_mean
            row['cv_status'] = cv_status
            annotated += 1

    with open(sweep_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in _CSV_FIELDNAMES})

    print(f'Annotated {annotated} row(s) in {sweep_file} with cv_macro_f1 / cv_status.')
    return annotated


def _print_revalidation_summary(cv_file, cv_pass_threshold, annotated_count):
    """Print PASS/REGRESSED tallies and the top revalidated configs."""
    if not os.path.exists(cv_file):
        return
    df = pd.read_csv(cv_file)
    if df.empty:
        return

    reval = df[df.get('source', '') == 'revalidation'] if 'source' in df.columns else df
    if reval.empty:
        return

    pass_count = int((reval['cv_status'] == 'PASS').sum())
    regressed_count = int((reval['cv_status'] == 'REGRESSED').sum())

    print('--- Revalidation summary ---')
    print(
        f'Total revalidated: {len(reval)} | '
        f'PASS (>= {cv_pass_threshold}): {pass_count} | '
        f'REGRESSED: {regressed_count} | '
        f'Sweep rows highlighted: {annotated_count}'
    )

    top_cols = [
        'cv_status', 'val_macro_f1', 'val_macro_f1_std',
        'original_val_macro_f1', 'delta_vs_original',
        'model', 'alpha', 'fit_prior', 'norm',
        'ngram_range', 'min_df', 'use_safe_text_cleaning',
    ]
    top_cols = [c for c in top_cols if c in reval.columns]
    top = reval.sort_values('val_macro_f1', ascending=False).head(5)[top_cols]
    print('Top 5 by CV mean:')
    print(top.to_string(index=False))
    print('----------------------------')


def main():
    data = load_data()
    run_sweep(data)


if __name__ == '__main__':
    main()
