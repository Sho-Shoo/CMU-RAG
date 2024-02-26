import os
from collections import deque
from rank_bm25 import BM25Okapi
from retriever.base_retriever import BaseRetriever


class BM25Retriever(BaseRetriever):

    def __init__(self, knowledge_source_path: str = "knowledge_source"):
        self.corpus = []
        file_names = os.listdir(knowledge_source_path)
        for file_name in file_names:
            with open(os.path.join(knowledge_source_path, file_name), "r") as file:
                doc = file.read().split("<sep>")
                self.corpus.extend(doc)

        self.corpus, tokenized_corpus = _process_corpus(self.corpus)
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, question: str, top_n: int = 5):
        tokenized_question = question.lower().split()
        return self.bm25.get_top_n(tokenized_question, self.corpus, n=top_n)


def _process_corpus(corpus: list[str]) -> tuple[list[str], list[list[str]]]:
    punctuations = "`~!@#$%^&*()_+[]\\;',./{}|:\"<>?"

    org_corpus_set = set()
    org_corpus = deque()
    tokenized_corpus = deque()
    for doc in corpus:
        if doc in org_corpus_set:
            continue

        d = doc.lower()
        for punc in punctuations:
            d = d.replace(punc, "")
        tokenized_doc = d.split()
        tokenized_corpus.append(tokenized_doc)
        org_corpus.append(doc)
        org_corpus_set.add(doc)

    return list(org_corpus), list(tokenized_corpus)


if __name__ == "__main__":
    r = BM25Retriever()
    docs = r.retrieve("Who is teaching 11711 in Spring 2024?")
    for doc in docs:
        print(doc)
        print()
