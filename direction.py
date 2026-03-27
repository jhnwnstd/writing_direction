"""
Directionality Analysis Script

Predicts writing direction (LTR vs RTL) from statistical analysis of character
frequency distributions at word-initial vs word-final positions.

Based on: Ashraf, M.I. & Sinha, S. (2018). PLoS ONE 13(1): e0190735.

Core principle: word beginnings universally use characters more diversely than
word endings. By comparing the entropy and inequality of character distributions
at the two ends of words, the method infers which end is the "beginning" and
thus which direction the script reads.

The method operates on STREAM ORDER — the first character in each word as it
appears in the input. For physical inscriptions or visual-order text, this
corresponds to the visual left terminal. For Unicode text, this corresponds to
the logical beginning (reading order), which differs from visual order for RTL
scripts.
"""

import argparse
import csv
import sys
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
from nltk.corpus import europarl_raw, udhr2
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

# (display_name, udhr2_fileid, script_family)
UDHR_LANGUAGES = [
    # RTL: Arabic script
    ("Arabic", "arb.txt", "Arabic"),
    ("Farsi", "pes_1.txt", "Arabic"),
    ("Urdu", "urd.txt", "Arabic"),
    ("Punjabi (Shah.)", "pnb.txt", "Arabic"),
    ("Saraiki", "skr.txt", "Arabic"),
    ("Uyghur", "uig_arab.txt", "Arabic"),
    ("Malay (Jawi)", "mly_arab.txt", "Arabic"),
    # RTL: Hebrew script
    ("Hebrew", "heb.txt", "Hebrew"),
    ("Yiddish", "ydd.txt", "Hebrew"),
    # RTL: Syriac script
    ("Assyrian", "aii.txt", "Syriac"),
    # RTL: Thaana script
    ("Dhivehi", "div.txt", "Thaana"),
    # LTR: Devanagari
    ("Hindi", "hin.txt", "Devanagari"),
    ("Bhojpuri", "bho.txt", "Devanagari"),
    ("Marathi", "mar.txt", "Devanagari"),
    ("Nepali", "nep.txt", "Devanagari"),
    ("Sanskrit", "san.txt", "Devanagari"),
    # LTR: other Indic
    ("Bengali", "ben.txt", "Bengali"),
    ("Punjabi (Gur.)", "pan.txt", "Gurmukhi"),
    ("Gujarati", "guj.txt", "Gujarati"),
    ("Kannada", "kan.txt", "Kannada"),
    ("Malayalam", "mal.txt", "Malayalam"),
    ("Tamil", "tam.txt", "Tamil"),
    # LTR: other scripts
    ("Armenian", "hye.txt", "Armenian"),
    ("Georgian", "kat.txt", "Georgian"),
    ("Korean", "kor.txt", "Hangul"),
    ("Myanmar", "mya.txt", "Myanmar"),
    ("Tigrinya", "tir.txt", "Ethiopic"),
    ("Khmer", "khm.txt", "Khmer"),
    ("Cree", "csw.txt", "Can. Syllabics"),
    ("Inuktitut", "ike.txt", "Can. Syllabics"),
    ("Vai", "vai.txt", "Vai"),
]

RESULT_FIELDNAMES = [
    "Language",
    "Script",
    "Words Analyzed",
    "Initial Gini",
    "Final Gini",
    "Delta Gini",
    "Initial Entropy",
    "Final Entropy",
    "Delta Entropy",
    "JSD",
    "Score",
    "Predicted",
    "Actual",
    "Correct",
]


def download_nltk_corpora() -> None:
    for name in ["europarl_raw", "udhr2"]:
        try:
            nltk.data.find(f"corpora/{name}")
        except LookupError:
            nltk.download(name, quiet=True)


def get_available_languages() -> List[str]:
    return [
        lang
        for lang in dir(europarl_raw)
        if lang.islower() and hasattr(getattr(europarl_raw, lang), "raw")
    ]


def ground_truth_direction(text: str) -> str:
    """Unicode bidi ground truth (validation only, not used in prediction)."""
    rtl = ltr = 0
    for ch in text:
        bidi = unicodedata.bidirectional(ch)
        if bidi in ("R", "AL"):
            rtl += 1
        elif bidi == "L":
            ltr += 1
    if rtl > ltr:
        return "RTL"
    if ltr > rtl:
        return "LTR"
    return "?"


# --- Core statistics ---


def gini_coefficient(freqs: np.ndarray) -> float:
    sorted_f = np.sort(freqs)
    n = len(sorted_f)
    if n == 0:
        return 0.0
    total = sorted_f.sum()
    if total == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * sorted_f)) / (n * total) - (n + 1) / n)


def shannon_entropy(freqs: np.ndarray) -> float:
    if len(freqs) == 0 or freqs.sum() == 0:
        return 0.0
    return float(entropy(freqs, base=2))


def sym_norm_diff(a: float, b: float) -> float:
    s = a + b
    return 2.0 * (a - b) / s if s else 0.0


def calc_jsd(p_counts: Counter, q_counts: Counter) -> float:
    chars = sorted(set(p_counts) | set(q_counts))
    if not chars:
        return 0.0
    p = np.array([p_counts.get(c, 0) for c in chars], dtype=float)
    q = np.array([q_counts.get(c, 0) for c in chars], dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)


# --- Text processing ---


def extract_words(text: str, delimiter: Optional[str] = None) -> List[str]:
    """
    Extract words from text. Each word is stripped of non-alphabetic
    characters (for handling punctuation in corpus text). Words with
    fewer than 2 characters are excluded since they contribute the
    same symbol to both positions.

    The delimiter parameter controls word boundary detection: None
    splits on whitespace (default for most corpora), or pass a
    specific string for other delimiters.
    """
    words = []
    for token in text.split(delimiter):
        word = "".join(ch for ch in token if ch.isalpha())
        if len(word) > 1:
            words.append(word)
    return words


def count_positional(words: List[str], n: int = 1) -> Tuple[Counter, Counter]:
    """Count first-n and last-n character n-grams from a word list."""
    first: Counter = Counter()
    last: Counter = Counter()
    for w in words:
        if len(w) > n:
            first[w[:n]] += 1
            last[w[-n:]] += 1
    return first, last


# --- Prediction ---


def compute_score(words: List[str]) -> Tuple[Dict[str, float], float]:
    """
    Compute directionality score from word list.

    Positive score → stream-initial end is more diverse (word beginnings) → LTR
    Negative score → stream-final end is more diverse (word beginnings) → RTL

    Analyzes unique word forms (types) rather than running-text tokens.
    This prevents high-frequency function words from dominating the
    positional distributions, following Ashraf & Sinha's methodology.

    The score is the symmetric normalized entropy difference between
    stream-initial and stream-final character distributions. Entropy
    alone outperforms combined Gini+entropy scoring because the Gini
    component introduces noise on small corpora.
    """
    types = list(set(words))
    initial, final = count_positional(types, n=1)
    init_f = np.array(list(initial.values()))
    fin_f = np.array(list(final.values()))

    ig, fg = gini_coefficient(init_f), gini_coefficient(fin_f)
    ie, fe = shannon_entropy(init_f), shannon_entropy(fin_f)
    dg = sym_norm_diff(ig, fg)
    de = sym_norm_diff(ie, fe)
    jsd = calc_jsd(initial, final)
    score = de  # entropy difference is the primary directional signal

    metrics = {
        "Initial Gini": ig,
        "Final Gini": fg,
        "Delta Gini": dg,
        "Initial Entropy": ie,
        "Final Entropy": fe,
        "Delta Entropy": de,
        "JSD": jsd,
    }
    return metrics, score


def predict_direction(score: float) -> str:
    if score > 0:
        return "LTR"
    if score < 0:
        return "RTL"
    return "?"


# --- Pipeline ---


def _collect_corpora(languages: List[str]) -> List[Tuple[str, str, str]]:
    corpora: List[Tuple[str, str, str]] = []
    for lang in languages:
        try:
            script = "Greek" if lang == "greek" else "Latin"
            corpora.append(
                (lang.capitalize(), getattr(europarl_raw, lang).raw(), script)
            )
        except Exception as e:
            print(f"  Skipping {lang}: {e}")
    for name, fileid, script in UDHR_LANGUAGES:
        try:
            corpora.append((name, udhr2.raw(fileid), script))
        except Exception as e:
            print(f"  Skipping {name}: {e}")
    return corpora


def process_languages(
    languages: List[str],
    sample_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    results = []
    for name, text, script_family in _collect_corpora(languages):
        print(f"  {name:<20s}", end="", flush=True)
        try:
            if sample_size is not None:
                cut = text[:sample_size]
                last_sp = cut.rfind(" ")
                text = cut[:last_sp] if last_sp > 0 else cut

            actual = ground_truth_direction(text)
            words = extract_words(text)

            # Convert to visual order for testing: RTL Unicode stores
            # words beginning-first (reading order), but visually the
            # beginning is on the RIGHT. Reverse each word so word[0]
            # = visual-left, word[-1] = visual-right — matching how an
            # unknown inscription would be scanned left-to-right.
            if actual == "RTL":
                words = [w[::-1] for w in words]

            metrics, score = compute_score(words)
            predicted = predict_direction(score)
            correct = "Y" if predicted == actual else "N"

            result = {
                "Language": name,
                "Script": script_family,
                "Words Analyzed": len(words),
                **metrics,
                "Score": score,
                "Predicted": predicted,
                "Actual": actual,
                "Correct": correct,
            }
            results.append(result)

            mark = "+" if correct == "Y" else "X"
            print(
                f"[{mark}] pred={predicted} actual={actual} score={score:+.4f}"
            )
        except Exception as e:
            print(f"ERROR: {e}")

    return results


# --- Output ---


def display_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No results to display.")
        return

    hdr = (
        f"{'Language':<20} {'Script':<16} {'Words':>6} "
        f"{'dG':>7} {'dS':>7} {'JSD':>6} {'Score':>7} "
        f"{'Pred':>4} {'True':>4} {'':>1}"
    )
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    correct = total = 0
    correct_ltr = total_ltr = 0
    correct_rtl = total_rtl = 0

    for r in results:
        mark = "+" if r["Correct"] == "Y" else "X"
        print(
            f"{r['Language']:<20} {r['Script']:<16} {r['Words Analyzed']:>6} "
            f"{r['Delta Gini']:>+7.3f} {r['Delta Entropy']:>+7.3f} "
            f"{r['JSD']:>6.3f} {r['Score']:>+7.4f} "
            f"{r['Predicted']:>4} {r['Actual']:>4} {mark:>1}"
        )
        total += 1
        if r["Correct"] == "Y":
            correct += 1
        if r["Actual"] == "LTR":
            total_ltr += 1
            if r["Correct"] == "Y":
                correct_ltr += 1
        elif r["Actual"] == "RTL":
            total_rtl += 1
            if r["Correct"] == "Y":
                correct_rtl += 1

    print("=" * len(hdr))
    print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    if total_ltr:
        print(f"  LTR: {correct_ltr}/{total_ltr}")
    if total_rtl:
        print(f"  RTL: {correct_rtl}/{total_rtl}")

    wrong = [r["Language"] for r in results if r["Correct"] == "N"]
    if wrong:
        print(f"  Wrong: {', '.join(wrong)}")


def save_results_to_csv(results: List[Dict[str, Any]], filename: str) -> None:
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore"
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    k: f"{v:.4f}" if isinstance(v, float) else str(v)
                    for k, v in r.items()
                }
            )
    print(f"Results saved to '{filename}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict writing direction from character frequency "
        "distributions at word-initial vs word-final positions."
    )
    parser.add_argument(
        "--save",
        "-s",
        metavar="FILE",
        nargs="?",
        const="directionality_results.csv",
        help="Save results to CSV",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Truncate corpus texts to this many characters",
    )
    args = parser.parse_args()

    download_nltk_corpora()
    languages = get_available_languages()
    if not languages:
        print("No languages found in 'europarl_raw' corpus.")
        sys.exit(1)

    n = len(languages) + len(UDHR_LANGUAGES)
    print(f"Analyzing {n} languages across 20 writing systems...\n")

    results = process_languages(languages, sample_size=args.sample_size)
    display_results(results)

    if args.save:
        save_results_to_csv(results, args.save)


if __name__ == "__main__":
    main()
