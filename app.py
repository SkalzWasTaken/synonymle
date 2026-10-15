import os
import random
from contextlib import redirect_stdout, redirect_stderr
from wordfreq import zipf_frequency, iter_wordlist
from AyDictionary import AyDictionary
from functools import lru_cache
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
dictionary = AyDictionary()

@lru_cache(maxsize=1)
def _get_common_words(lang, min_zipf, max_zipf):
    return [
        word for word in iter_wordlist(lang) 
        if min_zipf <= zipf_frequency(word, lang) <= max_zipf
        and word.isalpha()
    ]

def get_word(lang='en', min_zipf=4.0, max_zipf=8.0, min_synonyms=5):
    common_words = _get_common_words(lang, min_zipf, max_zipf)
    while True:
        candidate_word = random.choice(common_words)
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                try:
                    synonyms = dictionary.synonym(candidate_word)
                except Exception:
                    synonyms = None
        if synonyms and len(synonyms) >= min_synonyms:
            return {
                "word": candidate_word,
                "score": round(zipf_frequency(candidate_word, lang), 2),
                "synonym_count": len(synonyms),
                "synonyms": synonyms
            }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_word")
def api_get_word():
    word_data = get_word()
    return jsonify(word_data)

if __name__ == "__main__":
    app.run(debug=True)