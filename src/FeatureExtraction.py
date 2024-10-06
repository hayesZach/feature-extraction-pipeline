import collections
import math
import string
import os

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import cmudict, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

import nltk

nltk.download('cmudict')
nltk.download('stopwords')


def AverageSentenceLength(text) -> float:
    # Tokenize sentences and words
    num_sentences = len(sent_tokenize(text))
    num_words = len(word_tokenize(text))
    
    # Calculate ASL
    return num_words / num_sentences

def AverageSyllablesPerWord(text) -> float:
    import textstat
    
    words = word_tokenize(text)
    total_syllables = 0
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    for word in words:
        total_syllables += textstat.syllable_count(word)
    
    return total_syllables / total_words
    
    
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

def RemoveHeaderText(text):
    import re
    
    # The header of each book usually ends with '** anything **' or '*** anything ***' 
    pattern = r'\*{2,}\s*.*?\s*\*{2,}'    
    found = re.search(pattern, text)
    
    if found:
        header_end_pos = found.end()
        return text[header_end_pos:].strip()
    return text

def RemoveFooterText(text):
    import re
    
    # The footer usually starts with '*** anything ***'
    pattern = r'\*{3,}\s*.*?\s*\*{3,}'
    
    footer_matches = list(re.finditer(pattern, text))
    if footer_matches:
        last_match = footer_matches[-1]
        footer_start_pos = last_match.start()
        return text[:footer_start_pos].strip()
    return text
    

if __name__ == '__main__':
    current_directory = os.getcwd()
    print(f"Current Working Directory: {current_directory}")
    
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    print(f"Parent Directory: {parent_directory}")
    
    # load writing sample
    with open('datasets/ProjectGutenbergTop2000/books/1.txt', 'r', encoding='utf-8-sig') as file:
        text = file.read()
    
    # Remove header text
    text = RemoveHeaderText(text)
    # Remove footer text
    text = RemoveFooterText(text)
    
    print(FleschReadingEaseScore(text))
    