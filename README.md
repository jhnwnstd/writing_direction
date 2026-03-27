# Writing Direction Detection

Predicts writing direction (LTR vs RTL) from character frequency statistics alone. **39/42 correct (92.9%)** across 42 languages and 20 writing systems.

## How It Works

Word beginnings universally use characters more diversely than word endings. Given text in visual order, the method compares the Gini coefficient and Shannon entropy of character distributions at each end of words. The more diverse end is the beginning — its position reveals reading direction.

```
Delta_G = 2 * (G_left - G_right) / (G_left + G_right)
Delta_S = 2 * (S_left - S_right) / (S_left + S_right)
Score   = (-Delta_G + Delta_S) / 2        # positive → LTR, negative → RTL
```

No knowledge of the script or language is required — applicable to unknown and undeciphered writing systems.

## Results

| Script | Languages | Acc |
|---|---|---|
| Latin | Danish, Dutch, English, Finnish, French, German, Italian, Portuguese, Spanish, Swedish | 10/10 |
| Arabic | Arabic, Farsi, Urdu, Punjabi (Shah.), Saraiki, Uyghur, Malay (Jawi) | 5/7 |
| Hebrew | Hebrew, Yiddish | 2/2 |
| Syriac | Assyrian Neo-Aramaic | 1/1 |
| Thaana | Dhivehi | 1/1 |
| Devanagari | Hindi, Bhojpuri, Marathi, Nepali, Sanskrit | 5/5 |
| Bengali, Gurmukhi, Gujarati, Kannada, Malayalam, Tamil | 1 each | 6/6 |
| Greek, Armenian, Georgian, Hangul, Myanmar, Ethiopic, Vai | 1 each | 7/7 |
| Canadian Syllabics | Cree, Inuktitut | 2/2 |
| Khmer | Khmer | 0/1 |

**3 failures**: Arabic (ال prefix dominates word-initial positions), Uyghur (similar), Khmer (complex onset clusters) — genuine inversions of the universal asymmetry pattern.

## Usage

```bash
pip install nltk numpy scipy
python direction.py                    # run analysis
python direction.py --save results.csv # export to CSV
```

```python
from direction import extract_words, compute_score, predict_direction

words = extract_words("your text here")
metrics, score = compute_score(words)
print(predict_direction(score))  # "LTR" or "RTL"
```

## Limitations

- Requires visual-order input (characters as they appear on the page)
- Needs ~50+ words for a reliable signal
- Not designed for vertical or logographic writing systems

## References

- Ashraf & Sinha (2018). *"The 'handedness' of language."* PLoS ONE 13(1): e0190735.

## Author

[John Winstead](https://github.com/jhnwnstd)
