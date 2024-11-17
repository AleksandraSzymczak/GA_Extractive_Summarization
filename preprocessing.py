"Class for article preprocessing"
import string

import nltk
import numpy as np


class Text_preprocessor:

    def __init__(self, article) -> None:
        self.article = article
        self.vocabulary_weights = {}
        self.sentences = []
        self.vocabulary_weights_word = {}
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")

    def assign_random_weights_to_list(self, vocabulary_list):
        self.vocabulary_weights_word = {round(np.random.rand(), 2): word for word in vocabulary_list}

    def create_vocabulary(self):
        def normalize_text(article):
            article = article.lower()
            article = article.translate(str.maketrans("", "", string.punctuation))
            return article

        article_normalized = normalize_text(self.article)
        tokens = nltk.word_tokenize(article_normalized)
        stop_words = set(nltk.corpus.stopwords.words("english"))
        lemmatizer = nltk.stem.WordNetLemmatizer()
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        self.assign_random_weights_to_list(list(set(filtered_tokens)))

    def create_sentence_weights(self, sentences):
        sentence_weights = []
        for sentence in sentences:
            sentence_token = nltk.word_tokenize(sentence)
            sentence_weight = sum(
                [self.vocabulary_weights_word[word] for word in sentence_token if word in self.vocabulary_weights_word]
            )
            sentence_weight = round(sentence_weight / len(sentence_token), 2)
            sentence_weights.append(sentence_weight)
        return sentence_weights

    def tokenize_sentences(self):
        sentences = nltk.sent_tokenize(self.article, language="english")
        self.sentences = [sentence.replace("\n", " ") for sentence in sentences]
