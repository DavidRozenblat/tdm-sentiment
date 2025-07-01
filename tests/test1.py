"""
Model Comparison Script: Original vs. Optimized FinBERT

For each sample title, this script computes sentiment probabilities
using both the original and optimized models and displays the
results in a table.
"""
import sys
import time
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

# ─── Setup project path ─────────────────────────────────────────────────────
SRC_PATH = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/src/')
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from config import *

# ─── User settings ─────────────────────────────────────────────────────────
CORPUS_NAME = 'Newyork20042023'
NUM_SAMPLES = 100  # set to 0 for all

RESULTS_CSV = PROJECT_PATH / 'tests' / 'model_comparison_per_sample.csv'
input_path = RESULTS_PATH / CORPUS_NAME
# ─── Functions ─────────────────────────────────────────────────────────────
def load_dataset(folder: Path, num_samples: int):
    """
    Load title samples from chunk CSVs under `folder`.
    """
    try:
        csv_file = next(folder.glob('chunk_*.csv'))
        df = pd.read_csv(csv_file)
        if num_samples and num_samples < len(df):
            df = df.sample(num_samples, random_state=42)
        print(f"Loaded {len(df)} samples from {csv_file}")
    except Exception as e:
        # fallback examples
        print(f'can\'t get samples. error: {e}')
    return df['title'].tolist()


def load_models():
    """
    Initialize both sentiment analyzers and measure load time.
    Returns (orig_model, opt_model, load_times).
    """
    timings = {}
    t0 = time.time()
    orig = sentiment_score.TextAnalysis(str(FINBERT_LOCAL_MODEL_PATH))
    timings['original'] = time.time() - t0

    t1 = time.time()
    opt = sentiment_score.TextAnalysis(str(OPTIMIZED_FINBERT_MODEL_PATH))
    #opt = optimized_model1.OptimizedFinBERT(
        #local_model_path=str(OPTIMIZED_FINBERT_MODEL_PATH),
        #original_model_path=str(FINBERT_LOCAL_MODEL_PATH),
        #quantization_type='int8',
        #force_optimization=False
    #)
    timings['optimized'] = time.time() - t1
    return orig, opt, timings


def get_probs(text: str, analyzer):
    """Return sentiment dict for a single text."""
    try:
        return analyzer.get_sentiment_dict(text)
    except Exception as e:
        print(f"Error on text '{text[:30]}...': {e}")
        return {'positive': None, 'negative': None, 'neutral': None}


def compare_results(original_results, optimized_results):
    """
    Compute summary statistics comparing two lists of sentiment dicts.
    Returns a dict of total_samples, identical_outputs, identical_classifications,
    mean_diff and max_diff per label.
    """
    stats = {
        'total_samples': len(original_results),
        'identical_outputs': 0,
        'identical_classifications': 0,
        'mean_diff': {'positive': 0, 'negative': 0, 'neutral': 0},
        'max_diff': {'positive': 0, 'negative': 0, 'neutral': 0}
    }
    diffs = []
    for orig, opt in zip(original_results, optimized_results):
        if orig is None or opt is None:
            continue
        # identical outputs
        if all(abs(orig.get(k,0) - opt.get(k,0)) < 1e-5 for k in orig.keys()|opt.keys()):
            stats['identical_outputs'] += 1
        # classification agreement
        orig_cls = max(orig, key=orig.get)
        opt_cls = max(opt, key=opt.get)
        if orig_cls == opt_cls:
            stats['identical_classifications'] += 1
        # diffs per label
        sample_diff = {}
        for lbl in ['positive','negative','neutral']:
            d = abs(orig.get(lbl,0) - opt.get(lbl,0))
            sample_diff[lbl] = d
            stats['max_diff'][lbl] = max(stats['max_diff'][lbl], d)
        diffs.append(sample_diff)
    # compute mean diffs
    if diffs:
        for lbl in ['positive','negative','neutral']:
            stats['mean_diff'][lbl] = sum(d[lbl] for d in diffs)/len(diffs)
    return stats


def format_comparison_table(diff_stats, load_times):
    """
    Display a summary table of comparison metrics.
    """
    data = [
        ['Total samples',             diff_stats['total_samples']],
        ['Identical outputs',         diff_stats['identical_outputs']],
        ['Classification agreement %', f"{diff_stats['identical_classifications']/diff_stats['total_samples']*100:.2f}"],
        ['Mean diff - Positive',      f"{diff_stats['mean_diff']['positive']:.6f}"],
        ['Mean diff - Negative',      f"{diff_stats['mean_diff']['negative']:.6f}"],
        ['Mean diff - Neutral',       f"{diff_stats['mean_diff']['neutral']:.6f}"],
        ['Max diff - Positive',       f"{diff_stats['max_diff']['positive']:.6f}"],
        ['Max diff - Negative',       f"{diff_stats['max_diff']['negative']:.6f}"],
        ['Max diff - Neutral',        f"{diff_stats['max_diff']['neutral']:.6f}"],
        ['Load time (orig) sec',      f"{load_times['original']:.2f}"],
        ['Load time (opt) sec',       f"{load_times['optimized']:.2f}"]
    ]
    print('\n=== Summary Comparison Table ===')
    print(tabulate(data, headers=['Metric','Value'], tablefmt='pipe'))


def main():
    print(f"=== Comparing models on '{CORPUS_NAME}' ===")
    samples = load_dataset(input_path, NUM_SAMPLES)
    orig_model, opt_model, load_times = load_models()

    records = []
    orig_results, opt_results = [], []
    for title in tqdm(samples, desc='Analyzing titles'):
        o = get_probs(title, orig_model)
        p = get_probs(title, opt_model)
        orig_results.append(o)
        opt_results.append(p)
        records.append({
            'title': title,
            'orig_positive': o.get('positive'),
            'opt_positive':  p.get('positive'),
            'orig_negative': o.get('negative'),
            'opt_negative':  p.get('negative'),
            'orig_neutral':  o.get('neutral'),
            'opt_neutral':   p.get('neutral')
        })

    df = pd.DataFrame(records)
    print('\n=== Per-sample Sentiment Comparison ===')
    print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

    diff_stats = compare_results(orig_results, opt_results)
    format_comparison_table(diff_stats, load_times)

    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved detailed results to {RESULTS_CSV}")

if __name__ == '__main__':
    main()