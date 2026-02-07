import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# Extend common keywords
TEXTURE_KEYWORDS = ['silky', 'smooth', 'rough', 'soft', 'creamy', 'velvety', 'gritty', 'slimy', 'fuzzy']
PLACE_KEYWORDS = ['cafe', 'park', 'city', 'room', 'garden', 'forest', 'beach', 'restaurant', 'mall']
GENDER_KEYWORDS = {
    "male": ["male", "men", "man", "boy", "for him", "his"],
    "female": ["female", "women", "woman", "girl", "for her", "her"],
    "unisex": ["unisex", "for all", "everyone"]
}

def extract_descriptions(sentence):
    """
    Extract descriptive words, mood, texture, setting, common noun places, and gender from a sentence.
    
    Args:
        sentence (str): Input text/sentence.
        
    Returns:
        dict: Keys -> descriptive_words, mood, texture, setting, common_noun_places, gender
    """
    doc = nlp(sentence.lower())  # lowercase for keyword matching
    
    # 1. Descriptive words (adjectives & adverbs)
    descriptive_words = [token.text for token in doc if token.pos_ in ['ADJ', 'ADV']]
    
    # 2. Mood (adjectives related to feelings/emotions)
    mood = []
    for token in doc:
        if token.pos_ == 'ADJ' and any(child.lemma_ in ['mood', 'feeling', 'vibe', 'atmosphere'] for child in token.children):
            mood.append(token.text)
    for chunk in doc.noun_chunks:
        if any(word in chunk.text for word in ['mood', 'vibe', 'atmosphere', 'feeling']):
            mood.extend([tok.text for tok in chunk if tok.pos_ == 'ADJ'])
    
    # 3. Texture (keyword match)
    texture = [token.text for token in doc if token.text.lower() in TEXTURE_KEYWORDS]
    
    # 4. Setting (named entity locations + common noun places)
    setting = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]
    common_noun_places = [chunk.text for chunk in doc.noun_chunks if any(tok.text in PLACE_KEYWORDS for tok in chunk)]
    setting += common_noun_places
    
    # 5. Gender detection (unisex first)
    gender = None
    for g, keywords in GENDER_KEYWORDS.items():
        if any(kw in sentence.lower() for kw in keywords):
            gender = g
            break
    
    # Remove duplicates
    descriptive_words = list(set(descriptive_words))
    mood = list(set(mood))
    texture = list(set(texture))
    setting = list(set(setting))
    common_noun_places = list(set(common_noun_places))
    
    return {
        "descriptive_words": descriptive_words,
        "mood": mood,
        "texture": texture,
        "setting": setting,
        "common_noun_places": common_noun_places,
        "gender": gender
    }

