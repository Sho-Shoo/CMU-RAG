from abc import ABC


class BaseRetriever(ABC):

    def retrieve(self, question: str, top_n: int = 5):
        raise NotImplementedError
