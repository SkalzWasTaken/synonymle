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
        and word.islower()
        and len(word) > 1
        and ' ' not in word 
        and '-' not in word
    ]

def filter_synonyms(raw_synonyms, raw_antonyms, target_word):
    if not raw_synonyms:
        return []
    
    antonyms_set = {a.lower().strip() for a in raw_antonyms} if raw_antonyms else set()
    
    return [
        s.lower().strip() for s in raw_synonyms 
        if ' ' not in s 
        and '-' not in s 
        and s.lower().strip() != target_word.lower().strip() 
        and s.isalpha() 
        and s.islower()
        and len(s) > 1
        and s.lower().strip() not in antonyms_set
    ]

def get_word(lang='en', min_zipf=4.0, max_zipf=8.0, min_synonyms=5):
    common_words = _get_common_words(lang, min_zipf, max_zipf)
    while True:
        candidate_word = random.choice(common_words)
        
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                try:
                    raw_synonyms = dictionary.synonym(candidate_word)
                except Exception:
                    raw_synonyms = None
                
                try:
                    raw_antonyms = dictionary.antonym(candidate_word)
                except Exception:
                    raw_antonyms = None
        
        synonyms = filter_synonyms(raw_synonyms, raw_antonyms, candidate_word)
            
        if len(synonyms) >= min_synonyms:
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

@app.route("/check_guess", methods=["POST"])
def api_check_guess():
    data = request.get_json()
    guess = data.get("guess", "").strip().lower()
    target_word = data.get("target", "").strip().lower()
    
    if not guess.isalpha() or zipf_frequency(guess, 'en') == 0.0:
        return jsonify({"status": "not_a_word"})
        
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            try:
                raw_synonyms = dictionary.synonym(target_word)
            except Exception:
                raw_synonyms = None
            try:
                raw_antonyms = dictionary.antonym(target_word)
            except Exception:
                raw_antonyms = None
                
    synonyms = filter_synonyms(raw_synonyms, raw_antonyms, target_word)
    
    if guess in synonyms:
        return jsonify({"status": "correct", "synonyms": synonyms})
    else:
        return jsonify({"status": "incorrect_word", "synonyms": synonyms})

if __name__ == "__main__":
    app.run(debug=True)