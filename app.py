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
from nltk.stem import WordNetLemmatizer
from wordfreq import zipf_frequency, iter_wordlist
from functools import lru_cache
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

def to_base_form(word):
    """Converts a word to its base canonical form (verbs, nouns, adjectives)."""
    # Try verb lemmatization first, then noun, then adjective
    verb_base = lemmatizer.lemmatize(word, pos='v')
    if verb_base != word:
        return verb_base
    noun_base = lemmatizer.lemmatize(word, pos='n')
    if noun_base != word:
        return noun_base
    return lemmatizer.lemmatize(word, pos='a')

@lru_cache(maxsize=1)
def _get_candidate_pool():
    """Caches a broad pool of common words on startup."""
    pool = []
    for word in iter_wordlist('en'):
        score = zipf_frequency(word, 'en')
        if 4.0 <= score <= 7.5 and word.isalpha() and word.islower() and len(word) > 1 and ' ' not in word and '-' not in word:
            # Ensure the pool word itself is in its base form
            base_word = to_base_form(word)
            if base_word not in pool:
                pool.append(base_word)
        if len(pool) >= 5000:
            break
    return pool

def get_wordnet_relations(word):
    """Extracts clean synonyms and antonyms from WordNet lemmas."""
    synonyms = set()
    antonyms = set()
    
    # Ensure target word lookup is normalized
    base_target = to_base_form(word)
    
    for syn in wn.synsets(base_target):
        for lemma in syn.lemmas():
            name = lemma.name().lower().replace('_', ' ')
            if name.isalpha() and name.islower() and len(name) > 1:
                # Convert synonym to base form so "teachers" -> "teacher"
                synonyms.add(to_base_form(name))
                
            if lemma.antonyms():
                for ant in lemma.antonyms():
                    ant_name = ant.name().lower().replace('_', ' ')
                    if ant_name.isalpha() and ant_name.islower() and len(ant_name) > 1:
                        antonyms.add(to_base_form(ant_name))
                        
    return synonyms, antonyms

def filter_synonyms(target_word):
    """Filters raw WordNet relations, stripping out the target word and antonyms."""
    base_target = to_base_form(target_word)
    raw_syns, raw_ants = get_wordnet_relations(base_target)
    return [
        s for s in raw_syns 
        if s != base_target 
        and s not in raw_ants
    ]

def get_word(lang='en', min_synonyms=3):
    """Picks a random word from the cached pool in base form."""
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
            
    return {
        "word": "teach",
        "score": 5.5,
        "synonym_count": 3,
        "synonyms": ["instruct", "educate", "train"]
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
    
    # Validate real word
    if not guess.isalpha() or zipf_frequency(guess, 'en') == 0.0:
        return jsonify({"status": "not_a_word"})
        
    # Convert player's guess to base form for checking
    base_guess = to_base_form(guess)
    synonyms = filter_synonyms(target_word)
    
    if base_guess in synonyms or guess in synonyms:
        return jsonify({"status": "correct", "synonyms": synonyms})
    else:
        return jsonify({"status": "incorrect_word", "synonyms": synonyms})

if __name__ == "__main__":
    app.run(debug=True)