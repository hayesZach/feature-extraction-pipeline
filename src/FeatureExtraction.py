import collections
import math
import string
import os
import re, textstat

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import cmudict, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

import nltk

nltk.download('cmudict')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def remove_special_chars(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def AverageSentenceLength(text) -> float:
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

def AverageSyllablesPerWord(text) -> float:
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

# The FRE score is a number between 0 and 100, with higher scores indicating easier reading
def FleschReadingEaseScore(text) -> float:
    # FRE = 206.835 - (1.015 * ASL) - (84.6 * ASW)
    # FRE assumes that shorter words and sentences are easier to read and understand
    
    # 90-100: Very easy
    # 80-89:  Easy
    # 70-79:  Fairly easy
    # 60-69:  Standard
    # 50-59:  Fairly difficult
    # 30-49:  Difficult
    # 0-29:   Very confusing
    
    ASL = AverageSentenceLength(text)
    ASW = AverageSyllablesPerWord(text)
    
    return 206.835 - (1.015 * ASL) - (84.6 * ASW)

# Indicates the education level required to understand the text.
def FleschKincaidGradeLevel(text) -> float:
    ASL = AverageSentenceLength(text)
    ASW = AverageSyllablesPerWord(text)
    
    return 0.39 * ASL + 11.8 * ASW - 15.59

def GunningFogIndex(text) -> float:
    # Gradelevel = 0.4(ASL + PHW)
    # PHW: The percentage of hard words
    ASL = AverageSentenceLength(text)
    PHW = PercentageHardWords(text)
    return 0.4 * (ASL + PHW)

def PercentageHardWords(text) -> float:
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

def is_complex_word(word):
    if re.match(r'(.*ing|.*ed|.*es|.*ly)$', word):
        return False
    
    syllable_count = textstat.syllable_count(word)
    return syllable_count >= 3

def ShannonEntropy(text) -> float:
    from scipy.stats import entropy
    from collections import Counter
    
    words = word_tokenize(text)
    
    # remove special characters from each word
    cleaned_words = [remove_special_chars(word) for word in words if word.isalpha()]
    
    word_frequencies = Counter(cleaned_words)
    
    total_words = len(cleaned_words)
    if total_words == 0:
        return 0.0
    
    probabilities = []
    for word in word_frequencies:
        prob = word_frequencies[word] / total_words
        probabilities.append(prob)
    
    probabilities_array = np.array(probabilities)
    
    return entropy(probabilities_array, base=2)

def SimpsonsIndex(text) -> float:
    from collections import Counter
    
    words = word_tokenize(text)
    
    # remove special characters from each word
    cleaned_words = [remove_special_chars(word) for word in words if word.isalpha()]
    
    word_frequencies = Counter(cleaned_words)
    
    N = sum(word_frequencies.values())
    if N <= 1:
        return 0.0
    
    sum_n_i = 0
    for n_i in word_frequencies.values():
        sum_n_i += n_i * (n_i - 1)
        
    D = 1 - (sum_n_i / (N * (N - 1)))
    
    return D
    
def DaleChallReadability(text) -> float:
    return textstat.dale_chall_readability_score(text)

def RemoveHeaderText(text):
    # The header of each book usually ends with '** anything **' or '*** anything ***' 
    pattern = r'\*{2,}\s*.*?\s*\*{2,}'    
    found = re.search(pattern, text)
    
    if found:
        header_end_pos = found.end()
        return text[header_end_pos:].strip()
    return text

def RemoveFooterText(text):
    # The footer usually starts with '*** anything ***'
    pattern = r'\*{3,}\s*.*?\s*\*{3,}'
    
    footer_matches = list(re.finditer(pattern, text))
    if footer_matches:
        last_match = footer_matches[-1]
        footer_start_pos = last_match.start()
        return text[:footer_start_pos].strip()
    return text