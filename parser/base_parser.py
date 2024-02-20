import os
from abc import ABC


class BaseParser(ABC):

    def __init__(self, doc_max_len: int = 100, doc_min_len: int = 3, output_dir: str = "knowledge_source"):
        self.doc_max_len = doc_max_len
        self.doc_min_len = doc_min_len
        self.output_dir = output_dir
        # find largest file index
        files = os.listdir(output_dir)
        if len(files) == 0:
            self.curr_index = 0
        else:
            latest_file = max(files)
            latest_index = int(latest_file.replace(".txt", ""))
            self.curr_index = latest_index + 1

    def parse(self, url: str) -> None:
        raise NotImplementedError

    def _save_doc(self, doc: str) -> None:
        words = doc.split()
        if len(words) < self.doc_min_len:
            return
        elif self.doc_min_len <= len(words) <= self.doc_max_len:
            self._write_doc(" ".join(words))
        else:
            self._write_doc(" ".join(words[:self.doc_max_len]))
            self._save_doc(" ".join(words[self.doc_max_len:]))

    def _write_doc(self, doc: str) -> None:
        file_name = f"{self.curr_index}.txt"
        file_path = os.path.join(self.output_dir, file_name)
        with open(file_path, "w") as file:
            file.write(doc)
        self.curr_index += 1
