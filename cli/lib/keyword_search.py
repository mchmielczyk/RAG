from nltk.stem import PorterStemmer

BM25_K1 = 1.5
BM25_B = 0.75
import json
import math
import os
import pickle
import string
from collections import Counter


def load_movies(path: str) -> dict:
    with open(path, "r") as movies:
        movies_database = dict(json.load(movies))
    return movies_database


def stemming(original_words: list[str]) -> list[str]:
    stemmed = list()
    stemmer = PorterStemmer()
    for movie_token in original_words:
        stemmed.append(stemmer.stem(movie_token))
    return stemmed


def tokenize_text(input_text: str) -> list[str]:
    return input_text.split()


def tokenize_text_single(input_text: str) -> str:

    output_token = preprocessing_query(input_text)

    if len(output_token) != 1:
        raise Exception("token not single")

    return output_token[0]


def preprocessing_stop_words(original_words: list[str]) -> list[str]:
    punctuation_table = str.maketrans("", "", string.punctuation)
    preprocessed_stop_words = list()
    for word in original_words:
        word = word.lower()
        word = word.translate(punctuation_table)
        preprocessed_stop_words.append(word)
    return preprocessed_stop_words


def preprocessing_title(original_movie_title: str) -> list[str]:

    # processing
    movie_title_preprocessed = original_movie_title.lower()
    # punctuation
    punctuation_table = str.maketrans("", "", string.punctuation)
    movie_title_preprocessed = movie_title_preprocessed.translate(punctuation_table)
    # tokenization
    movie_title_preprocessed = tokenize_text(movie_title_preprocessed)

    movie_title_filtered = list()

    stop_words = open("data/stopwords.txt").read().splitlines()
    stop_words_preprocessed = preprocessing_stop_words(stop_words)
    for token in movie_title_preprocessed:
        if token not in stop_words_preprocessed:
            movie_title_filtered.append(token)

    movie_title_filtered = preprocessing_stop_words(movie_title_filtered)

    movie_title_filtered = stemming(movie_title_filtered)

    return movie_title_filtered


def preprocessing_query(original_query: str) -> list[str]:

    preprocessed_query = original_query.lower()

    preprocessed_query = tokenize_text(preprocessed_query)

    filtered_query = list()

    stop_words = open("data/stopwords.txt").read().splitlines()
    stop_words_preprocessed = preprocessing_stop_words(stop_words)
    for token in preprocessed_query:
        if token not in stop_words_preprocessed:
            filtered_query.append(token)

    filtered_query = preprocessing_stop_words(filtered_query)

    filtered_query = stemming(filtered_query)

    return filtered_query


class InvertedIndex:
    def __init__(self) -> None:
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies = dict()
        self.doc_lengths = dict()
        self.doc_lengths_path = os.path.join("cache", "doc_lengths.pkl")
        self.index_path = os.path.join("cache", "index.pkl")

    def __add_document(self, doc_id, text) -> None:
        if doc_id not in self.docmap:
            self.docmap[doc_id] = text

        token_list = preprocessing_title(f"{text['title']} {text['description']}")

        self.doc_lengths[doc_id] = len(token_list)

        cnt_tmp = Counter(token_list)
        self.term_frequencies[doc_id] = cnt_tmp

        for token in token_list:
            if token not in self.index:
                self.index[token] = []
            if not doc_id in self.index[token]:
                self.index[token].append(doc_id)

    def __get_avg_doc_length(self) -> float:
        avg_length = 0
        for doc_id, actual_doc_length in self.doc_lengths.items():
            avg_length += actual_doc_length

        if len(self.docmap) == 0:
            return 0
        else:
            return avg_length / len(self.docmap)

    def get_documents(self, term: str) -> dict:

        search_query = preprocessing_query(str(term))
        list_of_queries = list()

        for token in search_query:
            if token in self.index:
                list_of_queries = self.index[token]

        list_of_queries = list_of_queries[:5]
        list_of_movies = dict()
        for movie_id in list_of_queries:
            list_of_movies[movie_id] = self.docmap[movie_id]

        return list_of_movies

    def build(self) -> None:
        movies_database = load_movies("data/movies.json")
        for movie in movies_database["movies"]:
            self.__add_document(movie["id"], movie)

    def save(self):
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
        except:
            raise Exception("index file error")
        try:
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
        except:
            raise Exception("index file error")
        try:
            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
        except:
            raise Exception("index file error")
        try:
            with open("cache/doc_lengths.pkl", "rb") as f:
                self.doc_lengths = pickle.load(f)
        except:
            raise Exception("index file error")

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id in self.term_frequencies and term in self.term_frequencies[doc_id]:
            return self.term_frequencies[doc_id][term]
        else:
            return 0

    def get_idf(self, term: str) -> float:
        term_match_doc_count = 0
        for movie_id, counter in self.term_frequencies.items():
            if term in counter:
                term_match_doc_count += 1

        return math.log((len(self.docmap) + 1) / (term_match_doc_count + 1))

    def get_tfidf(self, doc_id, term) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        df = 0
        total_documents = len(self.docmap)
        for movie_id, counter in self.term_frequencies.items():
            if term in counter:
                df += 1

        return math.log((total_documents - df + 0.5) / (df + 0.5) + 1)

    def bm25(self, doc_id, term) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        raw_tf = self.get_tf(doc_id, term)
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    def bm25_search(self, query, limit):
        tokenized_query = preprocessing_query(query)
        scores = {doc_id: 0 for doc_id in self.docmap}
        for token in tokenized_query:
            for doc_id, tokens in self.docmap.items():
                scores[doc_id] += self.bm25(doc_id, token)

        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        return dict(list(scores.items())[:limit])
