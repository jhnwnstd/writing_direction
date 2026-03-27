# Writing Direction Detection

Predicts writing direction (LTR vs RTL) purely from statistical analysis of character frequency distributions at word-initial and word-final positions. Tested across **42 languages** and **20 writing systems** with **92.9% accuracy**.

Based on: Ashraf, M.I. & Sinha, S. (2018). *"The 'handedness' of language: Directional symmetry breaking of sign usage in words."* PLoS ONE 13(1): e0190735.

## Core Principle

Word beginnings universally use characters more diversely than word endings. This is a statistical property of human language — suffixes and word-final patterns are more restricted than the variety of ways words can begin.

By measuring the inequality (Gini coefficient) and diversity (Shannon entropy) of character distributions at each end of words, the method determines which end is the "beginning" and thus infers reading direction:

- **More diverse left terminal** → word beginnings are on the left → **Left-to-Right**
- **More diverse right terminal** → word beginnings are on the right → **Right-to-Left**

The method is language-agnostic and requires no knowledge of the script — only the positional character frequencies matter. This makes it applicable to unknown or undeciphered writing systems.

## Results

### Accuracy: 39/42 (92.9%)

| Script Family | Languages Tested | Accuracy |
|---|---|---|
| Latin | Danish, Dutch, English, Finnish, French, German, Italian, Portuguese, Spanish, Swedish | 10/10 |
| Greek | Greek | 1/1 |
| Arabic | Arabic, Farsi, Urdu, Punjabi (Shahmukhi), Saraiki, Uyghur, Malay (Jawi) | 5/7 |
| Hebrew | Hebrew, Yiddish | 2/2 |
| Syriac | Assyrian Neo-Aramaic | 1/1 |
| Thaana | Dhivehi | 1/1 |
| Devanagari | Hindi, Bhojpuri, Marathi, Nepali, Sanskrit | 5/5 |
| Bengali | Bengali | 1/1 |
| Gurmukhi | Punjabi (Gurmukhi) | 1/1 |
| Gujarati, Kannada, Malayalam, Tamil | 1 each | 4/4 |
| Armenian, Georgian, Hangul, Myanmar, Ethiopic | 1 each | 5/5 |
| Khmer | Khmer | 0/1 |
| Canadian Syllabics | Cree, Inuktitut | 2/2 |
| Vai | Vai | 1/1 |

### Known Failures (3/42)

- **Arabic**: The definite article *al-* (ال) dominates word-initial positions, inverting the typical diversity pattern
- **Uyghur**: Similar morphological concentration at word beginnings
- **Khmer**: Complex onset consonant clusters concentrate the initial position

These represent genuine exceptions to the universal asymmetry pattern — languages where word beginnings are *more* concentrated than word endings due to specific morphological or phonological properties.

## Methodology

### Statistical Metrics

For each text, the method extracts the first and last character of every word (after stripping punctuation) and computes:

1. **Gini Coefficient** — measures inequality in character frequency distributions (0 = perfectly equal, 1 = maximally unequal)
2. **Shannon Entropy** (base-2) — measures diversity of character usage (higher = more diverse)
3. **Jensen-Shannon Divergence** — measures how different the initial and final distributions are

### Scoring

The directionality score uses the Ashraf & Sinha symmetric normalized difference:

```
Delta_G = 2 * (G_initial - G_final) / (G_initial + G_final)
Delta_S = 2 * (S_initial - S_final) / (S_initial + S_final)
Score   = (-Delta_G + Delta_S) / 2
```

- **Positive score** → Left-to-Right
- **Negative score** → Right-to-Left
- **Magnitude** indicates strength of the directional signal

## Files

- **direction.py** — Main analysis script
- **directionality_results.csv** — Results across all 42 languages
- **The_handedness_of_language_Directional.pdf** — Original research paper
- **pyproject.toml** — Project configuration

## Installation

```bash
pip install nltk numpy scipy
```

NLTK corpora (`europarl_raw`, `udhr2`) are downloaded automatically on first run.

## Usage

### Run the full analysis

```bash
python direction.py
```

### Save results to CSV

```bash
python direction.py --save results.csv
```

### Analyze custom text

```python
from direction import extract_words, compute_score, predict_direction

words = extract_words("your text here with multiple words")
metrics, score = compute_score(words)
direction = predict_direction(score)
print(f"Direction: {direction}, Score: {score:+.4f}")
```

## Implications for Decipherment

The method's primary application is determining reading direction for unknown or undeciphered scripts. Given a corpus of text in visual order (as it appears on the page or inscription):

1. Extract word-groups separated by spaces or other delimiters
2. Take the leftmost and rightmost characters from each group
3. Compare their frequency distributions
4. The more diverse end is where words begin — that's the reading start direction

This requires no prior knowledge of the script, language, or encoding.

## Limitations

- Requires visual-order input (characters ordered as they appear on the page)
- Needs sufficient text (~50+ words for a signal, more for reliability)
- 3 known exceptions where morphological properties invert the universal pattern
- Not designed for vertical writing systems
- Not applicable to logographic systems without clear word boundaries

## License

MIT License

## Author

Created by [John Winstead](https://github.com/jhnwnstd)
