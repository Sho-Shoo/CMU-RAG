import boto3
import botocore.errorfactory
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel
import json
from retriever.bm25_retriever import BM25Retriever
from aws_config import AWSConfig
from retriever.base_retriever import BaseRetriever
from evaluation_metric.evaluation import f1_score, recall_score, exact_match_score, normalize_answer, write_test_result
from retriever.embedding_retriever import EmbeddingRetriever
from pprint import pp, pformat
from language_model.utils import get_in_context_example

FEW_TEMPLATE = \
"""
[INST]<<SYS>>
You are a question-answering assistant. You will be provided with a set of DOCUMENTS about Carnegie Mellon University (CMU) and the Language Technology Institute (LTI), marked by 'CONTEXT START' and 'CONTEXT END'. These DOCUMENTS are composed of multiple subdocuments, each separated by a newline character ('\n'). Your task is to provide a short and simple answer to the QUESTION based on the DOCUMENTS, considering only the most relevant subdocument. Begin your answer with a concise statement that directly addresses the question. If the DOCUMENTS do not contain the necessary information, please answer "I don't know". Keep the answer brief and to the point. Here are a few examples marked by 'EXAMPLE START' and 'EXAMPLE END':

EXAMPLE START 

{in-context learning}

EXAMPLE END

CONTEXT START

{context}

CONTEXT END
<</SYS>>[/INST]

[INST]
QUESTION: {question}
[\INST]

ANSWER:
"""

ZERO_TEMPLATE = \
"""
You are a question-answering assistant. You will be provided with a set of DOCUMENTS about Carnegie Mellon University (CMU) and the Language Technology Institute (LTI), marked by 'CONTEXT START' and 'CONTEXT END'. These DOCUMENTS are composed of multiple subdocuments, each separated by a newline character ('\n'). Your task is to provide a short and simple answer to the QUESTION based on the DOCUMENTS, considering only the most relevant subdocument. Begin your answer with a concise statement that directly addresses the question. If the DOCUMENTS do not contain the necessary information, please answer "I don't know". Keep the answer brief and to the point. Here are a few examples marked by 'EXAMPLE START' and 'EXAMPLE END':

EXAMPLE START 

{in-context learning}

EXAMPLE END

CONTEXT START

{context}

CONTEXT END
<</SYS>>[/INST]

[INST]
QUESTION: {question}
[\INST]

ANSWER:
"""


def _build_llama2_prompt(context: str, question: str, few_shot: bool = True) -> str:
    if few_shot:
        template = FEW_TEMPLATE
        template = template.replace("{in-context learning}", get_in_context_example())
    else:
        template = ZERO_TEMPLATE

    prompt = template.replace("{context}", context).replace("{question}", question)
    return prompt


class SageMakerLlama27B:

    def __init__(self, retriever: BaseRetriever, top_n: int = 5):
        """
        Initialize a Sagemaker endpoint that runs Llama2 7B
        :param retriever: document retriever
        :param top_n: top n documents that retriever should retrieve
        """
        self.retriever = retriever
        self.top_n = top_n
        SageMakerLlama27B.set_up()

    @classmethod
    def set_up(cls):
        role = AWSConfig.SAGEMAKER_ARN_ROLE
        llama_model = JumpStartModel(model_id="meta-textgeneration-llama-2-7b-f", role=role)
        print("Deploying Llama-2 7B to SageMaker...")
        llama_model.deploy(initial_instance_count=1,
                           instance_type="ml.g5.4xlarge",
                           endpoint_name=AWSConfig.SAGEMAKER_ENDPOINT_NAME,
                           accept_eula=True)

    @classmethod
    def shut_down(cls):
        """
        Shut down Sagemaker endpoint in destructor in order to save cost
        """
        print(f"Shutting down Sagemaker endpoint {AWSConfig.SAGEMAKER_ENDPOINT_NAME}...")
        sagemaker_session = sagemaker.Session()
        sagemaker_session.delete_endpoint(AWSConfig.SAGEMAKER_ENDPOINT_NAME)
        sagemaker_session.delete_endpoint_config(AWSConfig.SAGEMAKER_ENDPOINT_NAME)

    def prompt(self, question: str, max_new_tokens: int = 512, top_p: float = 0.9, temperature: float = 0.6,
               print_prompt=False):
        documents = self.retriever.retrieve(question, top_n=self.top_n)
        context = "\n".join(documents)
        prompt = _build_llama2_prompt(context, question)

        if print_prompt: print(prompt)

        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_new_tokens,
                           "top_p": top_p,
                           "temperature": temperature,
                           "return_full_text": True}
        }

        runtime = boto3.client("runtime.sagemaker")
        payload = json.dumps(payload, indent=4).encode('utf-8')
        response = runtime.invoke_endpoint(EndpointName=AWSConfig.SAGEMAKER_ENDPOINT_NAME,
                                           ContentType="application/json",
                                           Body=payload)
        return json.loads(response["Body"].read())

    @classmethod
    def prompt_without_initialization(cls, retriever: BaseRetriever, question: str, top_n: int = 5,
                                      max_new_tokens: int = 1024,  top_p: float = 0.9, temperature: float = 0.6,
                                      print_prompt=False, few_shot=False):
        if top_n == 0:
            raise RuntimeError("top_n is being reduced to 0.")

        documents = retriever.retrieve(question, top_n=top_n)
        context = "\n".join(documents)
        prompt = _build_llama2_prompt(context, question, few_shot=few_shot)

        if print_prompt: print(prompt)

        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_new_tokens,
                           "top_p": top_p,
                           "temperature": temperature,
                           "return_full_text": False}
        }

        runtime = boto3.client("runtime.sagemaker")
        payload = json.dumps(payload, indent=4).encode('utf-8')
        try:
            response = runtime.invoke_endpoint(EndpointName=AWSConfig.SAGEMAKER_ENDPOINT_NAME,
                                               ContentType="application/json",
                                               Body=payload)
        except botocore.errorfactory.ClientError:
            print(f"top_n = {top_n} exceeded max token limitation, decreasing top_n by 1 ...")
            return cls.prompt_without_initialization(retriever, question, top_n=top_n - 1, max_new_tokens=max_new_tokens,
                                                     top_p=top_p, temperature=temperature, print_prompt=print_prompt,
                                                     few_shot=few_shot)

        return json.loads(response["Body"].read())[0]["generated_text"]


def run_test():
    USE_BM25 = True
    FEW_SHOT = False
    TOP_N = 10 if USE_BM25 else 3

    # (USE_BM25, FEW_SHOT) => <file name>
    file_name_map = {
        (False, False): "llama2_embed_zero.txt",
        (False, True): "llama2_embed_few.txt",
        (True, False): "llama2_bm25_zero.txt",
        (True, True): "llama2_bm25_few.txt"
    }
    result_file_name = file_name_map[(USE_BM25, FEW_SHOT)]

    questions, answers, reference_answers, question_cates = [], [], [], []
    f1_scores, recall_scores, exact_match_scores, cate_summary = [], [], [], dict()

    with open("data/test/questions.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            questions.append(line.strip())

    with open("data/test/reference_answers.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            reference_answers.append(line.strip())

    with open("data/test/question_categories.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            question_cates.append(line.strip())

    retriever = BM25Retriever() if USE_BM25 else EmbeddingRetriever(TOP_N)
    # SageMakerLlama27B.set_up()

    for i, q in enumerate(questions):
        print(f"{i} / {len(questions)}")
        print(f"Q: {q}")
        ref_a = reference_answers[i]
        a = SageMakerLlama27B.prompt_without_initialization(retriever, q, top_n=TOP_N, print_prompt=False, few_shot=FEW_SHOT)
        answers.append(a)
        print(f"A: {a}")
        print("============================")

        f1 = f1_score(a, [ref_a], normalize_fn=normalize_answer)
        recall = recall_score(a, [ref_a], normalize_fn=normalize_answer)
        em = exact_match_score(a, [ref_a], normalize_fn=normalize_answer)

        f1_scores.append(f1)
        recall_scores.append(recall)
        exact_match_scores.append(em)

        category = question_cates[i]
        if category not in cate_summary:
            cate_summary[category] = {
                "f1": [],
                "recall": [],
                "em": []
            }
        cate_summary[category]["f1"].append(f1)
        cate_summary[category]["recall"].append(recall)
        cate_summary[category]["em"].append(em)

    for category in cate_summary:
        for metric in ["f1", "recall", "em"]:
            scores = cate_summary[category][metric]
            cate_summary[category][metric] = sum(scores) / len(scores)

    f1, recall, em = sum(f1_scores) / len(f1_scores), sum(recall_scores) / len(recall_scores), sum(exact_match_scores) / len(exact_match_scores)
    test_summary = f"F1 score: {f1}\n" + \
                   f"Recall score: {recall}\n" + \
                   f"EM score: {em}\n" + \
                   pformat(cate_summary)
    print(test_summary)
    write_test_result("data/test/" + result_file_name, answers, test_summary)

    SageMakerLlama27B.shut_down()


def run_submission():

    # SageMakerLlama27B.set_up()

    # TOP_N = 3
    # FEW_SHOT = False
    # RETRIEVER = EmbeddingRetriever(TOP_N)
    # OUTPUT_FILE = "data/submission/system_output_2.txt"

    TOP_N = 10
    FEW_SHOT = False
    RETRIEVER = BM25Retriever()
    OUTPUT_FILE = "data/submission/system_output_3.txt"

    questions = []
    with open("data/submission/questions.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            questions.append(line.strip())

    answers = []
    for i, q in enumerate(questions):
        print(f"{i} / {len(questions)}")
        print(f"Q: {q}")
        a = SageMakerLlama27B.prompt_without_initialization(RETRIEVER, q, top_n=TOP_N, print_prompt=False,
                                                            few_shot=FEW_SHOT)
        answers.append(a.strip().replace("\n", " ") + "\n")
        print(f"A: {a}")

    with open(OUTPUT_FILE, "w") as f:
        f.writelines(answers)

    # SageMakerLlama27B.shut_down()


if __name__ == "__main__":
    run_submission()
