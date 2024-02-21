import unstructured.documents.html
from parser.base_parser import BaseParser
from unstructured.partition.html import partition_html


def _merge_elements(elements: list) -> list[str]:
    groups = []
    curr_content_group = ""
    for elem in elements:
        if type(elem) == unstructured.documents.html.HTMLTitle:
            if curr_content_group:
                groups.append(curr_content_group)
                curr_content_group = ""
            curr_content_group += (elem.text + " ")
        else:
            curr_content_group += (elem.text + " ")

    if curr_content_group: groups.append(curr_content_group)  # add last group

    return groups


class HTMLParser(BaseParser):

    def parse(self) -> None:
        elems = partition_html(filename=self.url)
        content_groups = _merge_elements(elems)
        for group in content_groups:
            self._write_doc(group)
        self._save_file()


if __name__ == "__main__":
    parser = HTMLParser('/Users/shoutianze/Desktop/Schedule - Spring Carnival 2024.html')
    parser.parse()
