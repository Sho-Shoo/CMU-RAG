# The implementation of the evaluation metric is based on 
# Reference: https://github.com/facebookresearch/atlas/blob/main/src/evaluation.py
# with slightly modification to fit the project's need.

import string
from collections import Counter
from typing import Callable
import regex

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

# Calculate the number of same tokens between prediction and ground truth
def cal_num_same(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    return num_same

# Calculate the recall score between prediction and ground truth
def recall(prediction, ground_truth, normalize_fn):
    num_same = cal_num_same(prediction, ground_truth, normalize_fn)
    recall = num_same / len(normalize_fn(ground_truth).split())

    return recall

# Calculate the F1 score between prediction and ground truth
def f1(prediction, ground_truth, normalize_fn):
    eps = 1e-10
    num_same = cal_num_same(prediction, ground_truth, normalize_fn)
    precision = num_same / len(normalize_fn(prediction).split())
    recall = num_same / len(normalize_fn(ground_truth).split())

    f1 = (2 * precision * recall) / (precision + recall + eps)

    return f1

def f1_score(prediction: str, ground_truths: list, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])

def recall_score(prediction: str, ground_truths: list, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([recall(prediction, gt, normalize_fn) for gt in ground_truths])

def exact_match_score(prediction: str, ground_truths: list, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([em(prediction, gt, normalize_fn) for gt in ground_truths])

def write_test_result(filepath: str, answers: list[str], f1_score: float, recall_score: float, em_score: float):
    with open(filepath, "w") as f:
        for ans in answers:
            ans = ans.replace("\n", " ")
            f.write(ans.strip() + "\n")

        f.write("\n")
        f.write(f"F1 score: {f1_score}\n")
        f.write(f"Recall score: {recall_score}\n")
        f.write(f"EM score: {em_score}\n")


if __name__ == "__main__":
    pred_1 = "Professor Teruko Mitamura's research area includes Information Extraction, Summarization and Question Answering, Information Retrieval, Text Mining and Analytics, Language Technologies for Education, and Natural Language Processing and Computational Linguistics."
    pred_2 = "The text does not specify Professor Teruko Mitamura's research area, therefore I cannot answer this question."
    ground_truths = ["Information Extraction, Summarization and Question Answering, Information Retrieval, Text Mining and Analytics, Language Technologies for Education, Natural Language Processing and Computational Linguistic.", "Professor Teruko Mitamura's research interests include Information Extraction, Summarization and Question Answering, Information Retrieval, Text Mining and Analytics, Language Technologies for Education, and Natural Language Processing and Computational Linguistics."]

    print(f"The F1 score for pred 1 is {f1_score(pred_1, ground_truths, normalize_fn=normalize_answer)}")
    print(f"The F1 score for pred 2 is {f1_score(pred_2, ground_truths, normalize_fn=normalize_answer)}")

    print(f"The recall score for pred 1 is {recall_score(pred_1, ground_truths, normalize_fn=normalize_answer)}")
    print(f"The recall score for pred 2 is {recall_score(pred_2, ground_truths, normalize_fn=normalize_answer)}")

    print(f"The exact match score for pred 1 is {exact_match_score(pred_1, ground_truths, normalize_fn=normalize_answer)}")
    print(f"The exact match score for pred 2 is {exact_match_score(pred_2, ground_truths, normalize_fn=normalize_answer)}")