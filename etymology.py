import pickle
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


dict_latin, dict_german = pickle.load(open("counts.pickle", "rb"))


def calc_latin_german_word(word):
    cnt_latin, cnt_german = dict_latin.get(word, 0), dict_german.get(word, 0)
    total = cnt_latin + cnt_german
    if total == 0:
        return 0., 0.
    else:
        return (cnt_latin / total), (cnt_german / total)


# taken from http://agailloty.rbind.io/project/nlp_clean-text/
def clean_text(text):
    """
    This function takes as input a text on which several
    NLTK algorithms will be applied in order to preprocess it
    """
    if type(text) == str:
        tokens = word_tokenize(text)
    elif type(text) == list:
        tokens = text
    else:
        raise ValueError("Provide string or list of words")
    # Remove the punctuations
    tokens = [word for word in tokens if word.isalpha()]
    # Lower the tokens
    tokens = [word.lower() for word in tokens]
    # Remove stopword
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    # Lemmatize
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word, pos = "v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos = "n") for word in tokens]
    return tokens


def calc_latin_german(text, normalize = True):
    text = clean_text(text)
    sum_german, sum_latin = 0., 0.
    for word in text:
        upd_latin, upd_german = calc_latin_german_word(word)
        sum_latin += upd_latin
        sum_german += upd_german
    if normalize:
        total = sum_german + sum_latin
        if total < 1e-5:
            raise ValueError("Text is not in English or too small")
        return sum_latin / total, sum_german / total
    else:
        return sum_latin, sum_german


