import os
import random
import nltk

# Ensure NLTK data path points to Render's writable temporary directory
nltk.data.path.append('/tmp/nltk_data')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir='/tmp/nltk_data')
    nltk.download('omw-1.4', download_dir='/tmp/nltk_data')

from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency, iter_wordlist
from functools import lru_cache
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@lru_cache(maxsize=1)
def _get_candidate_pool():
    """Caches a broad, pre-filtered pool of common words on startup for instant, zero-lag gameplay."""
    pool = []
    for word in iter_wordlist('en'):
        score = zipf_frequency(word, 'en')
        if 4.0 <= score <= 7.5 and word.isalpha() and word.islower() and len(word) > 1 and ' ' not in word and '-' not in word:
            pool.append(word)
        if len(pool) >= 10000:  # Large enough for massive variety, small enough to initialize instantly
            break
    return pool

def get_wordnet_relations(word):
    """Extracts and normalizes clean synonyms and antonyms from WordNet lemmas."""
    synonyms = set()
    antonyms = set()
    
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().lower().replace('_', ' ')
            if name.isalpha() and name.islower() and len(name) > 1:
                synonyms.add(name)
                
            if lemma.antonyms():
                for ant in lemma.antonyms():
                    ant_name = ant.name().lower().replace('_', ' ')
                    if ant_name.isalpha() and ant_name.islower() and len(ant_name) > 1:
                        antonyms.add(ant_name)
                        
    return synonyms, antonyms

def filter_synonyms(target_word):
    """Filters raw WordNet relations, stripping out the target word and antonyms."""
    raw_syns, raw_ants = get_wordnet_relations(target_word)
    return [
        s for s in raw_syns 
        if s != target_word.lower() 
        and s not in raw_ants
    ]

def get_word(lang='en', min_synonyms=3):
    """Picks a random word from the cached pool that satisfies the minimum synonym threshold."""
    candidates = list(_get_candidate_pool())
    random.shuffle(candidates)
    
    for candidate_word in candidates:
        synonyms = filter_synonyms(candidate_word)
        if len(synonyms) >= min_synonyms:
            return {
                "word": candidate_word,
                "score": round(zipf_frequency(candidate_word, lang), 2),
                "synonym_count": len(synonyms),
                "synonyms": synonyms
            }
            
    # Fallback default if candidate scan exhausts
    return {
        "word": "happy",
        "score": 5.5,
        "synonym_count": 3,
        "synonyms": ["glad", "cheerful", "joyful"]
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
    
    # Validate if the guess is an actual recognized word in English
    if not guess.isalpha() or zipf_frequency(guess, 'en') == 0.0:
        return jsonify({"status": "not_a_word"})
        
    synonyms = filter_synonyms(target_word)
    
    if guess in synonyms:
        return jsonify({"status": "correct", "synonyms": synonyms})
    else:
        return jsonify({"status": "incorrect_word", "synonyms": synonyms})

if __name__ == "__main__":
    app.run(debug=True)