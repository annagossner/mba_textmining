import csv
import string
import re
import os
import nltk
from nltk import pos_tag
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from autocorrect import spell
from nltk.corpus import wordnet
import pandas as pd
from nltk.stem import WordNetLemmatizer
from collections import Counter,OrderedDict
from pandas import DataFrame
from afinn import Afinn


#IN: string, boolean (stem yes or no)
#OUT: list of tokens
def get_clean_tokens(text, dostem):
    text = filter_nonprintable(text)
    text = lowercase(text)
    text = strip_punctuation(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    if dostem:
        tokens = lemmatize(tokens)
        tokens = stem(tokens)

    return tokens

#IN: list of tokens and int number of desired terms
#OUT: prints, but returns None
def print_term_distribution(tokens,n):
    termcounter = Counter(tokens)
    highest = OrderedDict(termcounter.most_common(n))
    lowest = OrderedDict(termcounter.most_common()[:-n - 1:-1])

    print('Top:===================')
    for t, c in highest.items():
        print('%s\t%i' % (t, c))
    print('Bottom:===================')
    for t, c in lowest.items():
        print('%s\t%i' % (t, c))

def get_top_terms(tokens, n):
    termcounter = Counter(tokens)
    termcounts = OrderedDict(termcounter.most_common(n))
    return termcounts

def get_afinn_sentiment(text):
    afinn = Afinn()
    return afinn.score(text)

#IN: directory with text files
#OUT: list of strings
def get_doc_list(dir):
    docs = []
    for filename in os.listdir(dir):
        if filename.endswith('.txt'):
            with open(os.path.join(dir, filename), 'r') as f:
                text = f.read().strip()
                text = text.replace('\n', ' ').replace('\t', ' ')
                docs.append(text)
    return docs

#IN string text
#OUT string text
def filter_nonprintable(text):
    import string
    # Get the difference of all ASCII characters from the set of printable characters
    nonprintable = set([chr(i) for i in range(128)]).difference(string.printable)
    # Use translate to remove all non-printable characters
    text = text.translate({ord(character):None for character in nonprintable})
    #remove newlines
    text = text.rstrip('\r\n').replace('\n', '')
    return text
#IN string text
#OUT string text
def strip_punctuation(text):
    text = ''.join([ l for l in text if (l not in string.punctuation)])
    #curly quotes
    text = text.replace(u'\u201c', '').replace(u'\u201d', '').replace(u'\u2019', '')
    return text

#IN string text
#OUT string text
def lowercase(text):
    return text.lower()

#IN string
#OUT string
def spellcheck(token):
    return spell(token)

#IN string text
#OUT list of words
def tokenize_text(text):
    return word_tokenize(text)

#IN list of words
#OUT list of words
def stem(tokens):
    porter = PorterStemmer()
    tokens = [porter.stem(t) for t in tokens]
    return tokens

#IN list of words
#OUT list of words
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for t in tokens:
        if t:
            lemmatized_tokens.append(
                lemmatizer.lemmatize(t, get_wordnet_pos(t)))

    return lemmatized_tokens
#IN list
#OUT list
def remove_stopwords(tokens):
    stop_words = stopwords.words('english')
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

def print_doc_term_matrix(doc_list, filename):
    vec = CountVectorizer(stop_words=stopwords.words('english'))
    #vec = TfidfVectorizer(stop_words=stopwords.words('english'), smooth_idf=False)
    X = vec.fit_transform(doc_list)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    df.to_csv(filename, sep='\t', quoting=csv.QUOTE_NONE)

def get_nouns(tokens):
    nouns = []
    tags = pos_tag(tokens)
    for tag in tags:
        if tag[1] == 'NN':
            nouns.append(tag[0])
    return nouns

def get_verbs(tokens):
    verbs = []
    tags = pos_tag(tokens)
    for tag in tags:
        if tag[1].startswith('VB'):
            verbs.append(tag[0])
    return verbs

def get_adjectives(tokens):
    adjectives = []
    tags = pos_tag(tokens)
    for tag in tags:
        if tag[1].startswith('JJ'):
            adjectives.append(tag[0])
    return adjectives

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def get_synonyms_and_antonyms(word):
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    return set(synonyms), set(antonyms)

def get_tense_score(tokens):
    score = 0
    verb_count = 0
    # -1 for each past tense verb
    # +1 for each present tense verb
    # total is divided by the total number of verbs
    pos_tagged_tokens = pos_tag(tokens)

    for token in pos_tagged_tokens:
        treebank_pos = token[1]
        if (treebank_pos.startswith('VB')):
            verb_count += 1
            if treebank_pos.endswith('D') or treebank_pos.endswith('N'):
                score -= 1
            else:
                score += 1

    return score/verb_count