import os
from abc import ABC


class BaseParser(ABC):

    def __init__(self, url: str, doc_max_len: int = 100, doc_min_len: int = 3, output_dir: str = "knowledge_source"):
        self.doc_max_len = doc_max_len # 500 characters max
        self.doc_min_len = doc_min_len
        self.output_dir = output_dir
        self.url = url
        self.content = ""


    def parse(self) -> None:
        raise NotImplementedError

    def _save_file(self) -> None:
        escaped_url = self.url.replace('/', '|')
        with open(os.path.join(self.output_dir, f"{escaped_url}.txt"), "w") as file:
            file.write(self.content)

    def _write_doc(self, doc: str, words: list = None) -> None:
        if not words: words = doc.split()
        if len(words) > self.doc_max_len:
            saved_words = words[:self.doc_max_len]
            self.content += " ".join(saved_words)
            self.content += "<sep>"
            self._write_doc("", words=words[self.doc_max_len:])
        else:
            saved_words = words[:self.doc_max_len]
            self.content += " ".join(saved_words)
            self.content += "<sep>"
