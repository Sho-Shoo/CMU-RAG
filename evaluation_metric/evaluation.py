# The implementation of the evaluation metric is based on 
# Reference: https://github.com/facebookresearch/atlas/blob/main/src/evaluation.py
# with slightly modification to fit the project's need.

import logging
import string
from collections import Counter
from typing import Callable

import regex

logger = logging.getLogger(__name__)

# Normalization and score functions from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Calculate the exact match(em) score between prediction and ground truth
def em(prediction, ground_truth, normalize_fn):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))

# Calculate the F1 score between prediction and ground truth
def f1(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])


def exact_match_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([em(prediction, gt, normalize_fn) for gt in ground_truths])


if __name__ == "__main__":
    pred_1 = "Professor Teruko Mitamura's research area is Information Extraction, Summarization and Question Answering, Information Retrieval, Text Mining and Analytics, Language Technologies for Education, and Natural Language Processing and Computational Linguistics."
    pred_2 = "The text does not specify Professor Teruko Mitamura's research area, therefore I cannot answer this question."
    ground_truths = ["Professor Teruko Mitamura's research area focuses on Information Extraction, Summarization and Question Answering, Information Retrieval, Text Mining and Analytics, Language Technologies for Education, Natural Language Processing and Computational Linguistic."]

    print(f1_score(pred_1, ground_truths, normalize_fn=normalize_answer))
    print(f1_score(pred_2, ground_truths, normalize_fn=normalize_answer))
    # print(exact_match_score(prediction, ground_truths, normalize_fn=normalize_answer))