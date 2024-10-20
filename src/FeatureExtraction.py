from helper import remove_special_chars, percentage_hard_words, average_sentence_length, average_syllables_per_word

import textstat
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


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
    
    ASL = average_sentence_length(text)
    ASW = average_syllables_per_word(text)
    
    return 206.835 - (1.015 * ASL) - (84.6 * ASW)

# Indicates the education level required to understand the text.
def FleschKincaidGradeLevel(text) -> float:
    ASL = average_sentence_length(text)
    ASW = average_syllables_per_word(text)
    
    return 0.39 * ASL + 11.8 * ASW - 15.59

def GunningFogIndex(text) -> float:
    # Gradelevel = 0.4(ASL + PHW)
    # PHW: The percentage of hard words
    ASL = average_sentence_length(text)
    PHW = percentage_hard_words(text)
    return 0.4 * (ASL + PHW)

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