import os
import random
import nltk

# Set NLTK data path to /tmp so it's fully writable on Render container deployments
nltk.data.path.append('/tmp/nltk_data')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir='/tmp/nltk_data')
    nltk.download('omw-1.4', download_dir='/tmp/nltk_data')

from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency, iter_wordlist
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def get_wordnet_relations(word):
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
    raw_syns, raw_ants = get_wordnet_relations(target_word)
    return [
        s for s in raw_syns 
        if s != target_word.lower() 
        and s not in raw_ants
    ]

# Build a large, randomized word pool on startup
VALID_WORD_POOL = []

def initialize_word_pool():
    global VALID_WORD_POOL
    print("Building randomized word pool...")
    
    # Collect a broad sample of common words from wordfreq
    candidates = []
    for word in iter_wordlist('en'):
        score = zipf_frequency(word, 'en')
        if 4.0 <= score <= 7.5 and word.isalpha() and word.islower() and len(word) > 1 and ' ' not in word and '-' not in word:
            candidates.append(word)
        # Limit initial collection to keep startup fast, but large enough for great variety
        if len(candidates) >= 1500:
            break
            
    # Shuffle the candidate list so we don't just take the top alphabetical words
    random.shuffle(candidates)
    
    # Filter a subset that guarantees at least 3 valid WordNet synonyms
    for word in candidates:
        syns = filter_synonyms(word)
        if len(syns) >= 3:
            VALID_WORD_POOL.append(word)
        if len(VALID_WORD_POOL) >= 300: # Maintain a diverse pool of 300+ valid words
            break
            
    if not VALID_WORD_POOL:
        VALID_WORD_POOL = ["happy", "fast", "bright", "smart", "brave", "calm", "cold", "dark"]
    print(f"Randomized pool ready with {len(VALID_WORD_POOL)} words.")

# Run initialization on boot
initialize_word_pool()

def get_word(lang='en'):
    candidate_word = random.choice(VALID_WORD_POOL)
    synonyms = filter_synonyms(candidate_word)
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