import boto3
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel
import json
from retriever.bm25_retriever import BM25Retriever
from aws_config import AWSConfig
from retriever.base_retriever import BaseRetriever
from evaluation_metric.evaluation import f1_score, recall_score, exact_match_score, normalize_answer, write_test_result

FEW_TEMPLATE = \
"""
[INST]<<SYS>>
You are a question-answering assistant who provides a short answer to a QUESTION based on the CONTEXT about Carnegie Mellon University (CMU) and Language Technology Institute (LTI). If the CONTEXT does not contain necessary information, please answer 'I don't know'. Please keep the answer short and simple. Here are a few examples:

Question: When is 2024 Spring Carnival?
Answer: April 11 to April 14.

Question: When was Carnegie Mellon University founded?
Answer: Year 1900.

CONTEXT:
{context}
<</SYS>>[/INST]

[INST]
QUESTION: {question}
[\INST]

ANSWER:
"""

ZERO_TEMPLATE = \
"""
[INST]<<SYS>>
You are a question-answering assistant who provides a short answer to a QUESTION based on the CONTEXT about Carnegie Mellon University (CMU) and Language Technology Institute (LTI). If the CONTEXT does not contain necessary information, please answer 'I don't know'. Please keep the answer short and simple.

CONTEXT:
{context}
<</SYS>>[/INST]

[INST]
QUESTION: {question}
[\INST]

ANSWER:
"""


def _build_llama2_prompt(context: str, question: str, few_shot: bool = True) -> str:
    if few_shot: template = FEW_TEMPLATE
    else: template = ZERO_TEMPLATE

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
        response = runtime.invoke_endpoint(EndpointName=AWSConfig.SAGEMAKER_ENDPOINT_NAME,
                                           ContentType="application/json",
                                           Body=payload)
        return json.loads(response["Body"].read())[0]["generated_text"]


if __name__ == "__main__":
    FEW_SHOT = True

    result_file_name = "llama2_bm25_few.txt" if FEW_SHOT else "llama2_bm25_zero.txt"
    questions, answers, reference_answers = [], [], []
    f1_scores, recall_scores, exact_match_scores = [], [], []

    with open("data/test/questions.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            questions.append(line)

    with open("data/test/reference_answers.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            reference_answers.append(line)

    retriever = BM25Retriever()
    # SageMakerLlama27B.set_up()

    for i, q in enumerate(questions):
        print(f"{i} / {len(questions)}")
        print(f"Q: {q}")
        ref_a = reference_answers[i]
        a = SageMakerLlama27B.prompt_without_initialization(retriever, q, top_n=10, print_prompt=False, few_shot=False)
        answers.append(a)
        print(f"A: {a}")
        print("============================")

        f1_scores.append(f1_score(a, [ref_a], normalize_fn=normalize_answer))
        recall_scores.append(recall_score(a, [ref_a], normalize_fn=normalize_answer))
        exact_match_scores.append(exact_match_score(a, [ref_a], normalize_fn=normalize_answer))

    f1, recall, em = sum(f1_scores) / len(f1_scores), sum(recall_scores) / len(recall_scores), sum(exact_match_scores) / len(exact_match_scores)
    test_summary = f"F1 score: {f1}\n" + \
                   f"Recall score: {recall}\n" + \
                   f"EM score: {em}"
    print(test_summary)
    write_test_result("data/test/" + result_file_name, answers, f1, recall, em)

    SageMakerLlama27B.shut_down()
