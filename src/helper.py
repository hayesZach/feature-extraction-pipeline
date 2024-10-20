import re, textstat
from nltk.tokenize import word_tokenize, sent_tokenize

def remove_special_chars(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def is_complex_word(word):
    if re.match(r'(.*ing|.*ed|.*es|.*ly)$', word):
        return False
    
    syllable_count = textstat.syllable_count(word)
    return syllable_count >= 3

def percentage_hard_words(text) -> float:
    words = word_tokenize(text)
    
    # remove special characters from each word
    cleaned_words = [remove_special_chars(word) for word in words if word.isalpha()]
    
    if len(cleaned_words) == 0:
        return 0.0
    
    hard_word_count = 0
    for word in cleaned_words:
        if is_complex_word(word):
            hard_word_count += 1
    
    return (hard_word_count / len(cleaned_words)) * 100

def average_sentence_length(text) -> float:
    # Tokenize sentences
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    
    # Tokenize words
    words = word_tokenize(text)
    
    # remove special characters from each word
    cleaned_words = [remove_special_chars(word) for word in words if word.isalpha()]
    num_words = len(cleaned_words)
    
    if num_sentences == 0 or num_words == 0:
        return 0.0
    
    # Calculate ASL
    return num_words / num_sentences

def average_syllables_per_word(text) -> float:
    words = word_tokenize(text)
    
    # remove special characters from each word
    cleaned_words = [remove_special_chars(word) for word in words if word.isalpha()]
    
    total_syllables = 0
    total_words = len(cleaned_words)
    
    if total_words == 0:
        return 0.0
    
    for word in cleaned_words:
        total_syllables += textstat.syllable_count(word)
    
    return total_syllables / total_words

def remove_header_text(text):
    # The header of each book usually ends with '** anything **' or '*** anything ***' 
    pattern = r'\*{2,}\s*.*?\s*\*{2,}'    
    found = re.search(pattern, text)
    
    if found:
        header_end_pos = found.end()
        return text[header_end_pos:].strip()
    return text

def remove_footer_text(text):
    # The footer usually starts with '*** anything ***'
    pattern = r'\*{3,}\s*.*?\s*\*{3,}'
    
    footer_matches = list(re.finditer(pattern, text))
    if footer_matches:
        last_match = footer_matches[-1]
        footer_start_pos = last_match.start()
        return text[:footer_start_pos].strip()
    return text