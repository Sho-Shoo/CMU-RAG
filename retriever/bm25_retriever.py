import os
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, knowledge_source_path: str = "knowledge_source"):
        self.corpus = []
        file_names = os.listdir(knowledge_source_path)
        for file_name in file_names:
            with open(os.path.join(knowledge_source_path, file_name), "r") as file:
                self.corpus.append(file.read().lower())

        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, question: str):
        tokenized_question = question.lower().split()
        return self.bm25.get_top_n(question, self.corpus)

r = BM25Retriever()
docs = r.retrieve("what masters program does LTI offer")
for doc in docs:
    print(doc)
    print()
