import os
import random
import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency, iter_wordlist
from functools import lru_cache
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Ensure NLTK WordNet data is available locally/on container build
try:
    wn.synsets('test')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

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

def get_wordnet_relations(word):
    synonyms = set()
    antonyms = set()
    
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().lower().replace('_', ' ')
            # Ensure single word, alphabetic, lowercase, no special formatting
            if name.isalpha() and name.islower() and len(name) > 1:
                synonyms.add(name)
                
            if lemma.antonyms():
                for ant in lemma.antonyms():
                    ant_name = ant.name().lower().replace('_', ' ')
                    if ant_name.isalpha() and ant_name.islower() and len(ant_name) > 1:
                        antonyms.add(ant_name)
                        
    return synonyms, antonyms

def filter_synonyms(target_word):
    raw_syns, raw_ants = get_wordnet_relations(target_word)
    
    # Remove the target word itself and any explicit antonyms
    valid_synonyms = [
        s for s in raw_syns 
        if s != target_word.lower() 
        and s not in raw_ants
    ]
    return valid_synonyms

def get_word(lang='en', min_zipf=4.0, max_zipf=8.0, min_synonyms=5):
    common_words = _get_common_words(lang, min_zipf, max_zipf)
    while True:
        candidate_word = random.choice(common_words)
        synonyms = filter_synonyms(candidate_word)
            
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
        
    synonyms = filter_synonyms(target_word)
    
    if guess in synonyms:
        return jsonify({"status": "correct", "synonyms": synonyms})
    else:
        return jsonify({"status": "incorrect_word", "synonyms": synonyms})

if __name__ == "__main__":
    app.run(debug=True)