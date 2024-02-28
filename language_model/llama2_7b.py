import boto3
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel
import json
from retriever.bm25_retriever import BM25Retriever
from aws_config import AWSConfig
from retriever.base_retriever import BaseRetriever


def _build_llama2_prompt(context: str, question: str, few_shot: bool = True) -> str:
    zero_templ = "[INST]<<SYS>>\nYou are a question-answering assistant that provides a short ANSWER to a QUESTION " \
               "based on the given CONTEXT about Carnegie Mellon University. If you do not know the answer and the " \
               "CONTEXT doesn't contain the answer truthfully say \"I don't know\". Please keep the answer short and " \
               "straight-forward. \n\nCONTEXT:\n{context}\n<</SYS>>[/INST]\n\n[INST]QUESTION: {question}[\INST]\n\nANSWER:"

    few_templ = "[INST]<<SYS>>\nYou are a question-answering assistant that provides a short ANSWER to a QUESTION " \
               "based on the given CONTEXT about Carnegie Mellon University. For example:\nWhere is CMU located?\n" \
               "Pittsburgh.\nWhen does 2024 Spring Carnival begin?\nApril 11th.\n\nIf you do not know the answer and the " \
               "CONTEXT doesn't contain the answer truthfully say \"I don't know\". Please keep the answer short and " \
               "straight-forward. \n\nCONTEXT:\n{context}\n<</SYS>>[/INST]\n\n[INST]QUESTION: {question}[\INST]\n\nANSWER:"
    if few_shot: template = few_templ
    else: template = zero_templ

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
    questions = [
        "When was Carnegie Mellon University founded?",
        "Who is the president of CMU?",
        "What are the research interests of Graham Neubig?",
        "What is the mascot of CMU?",
        "What courses are offered by Graham Neubig at CMU?",
        "Who teaches 11711 in Spring 2024?",
        "Which CMU grad student developed Java?",
        "When is 2024 Spring Carnival?",
        "How many students does CMU have?",
        "Who taught 11711 Advanced Natural Language Processing in Fall 2023?",
        "Who will teach 11742 in Fall 2024?",
        "What are the Master programs in LTI?",
        "When was the first freshman-level computer science course launched in CMU?",
        "When was the Robotics Institute started?",
        "What is Professor Teruko Mitamura's research area?",
        "Where is Professor Graham Neubig's office?",
        "Who are the authors of the paper 'Generalized Glossing Guidelines: An Explicit, Human- and Machine-Readable, Item-and-Process Convention for Morphological Annotation'?",
        "Who is teaching Multimodal Machine Learning in Spring 2024?",
        "Who is teaching 11711 in Fall 2023?",
        "When is CMU Spring Break in 2024?",
        "When is the semester drop deadline for Fall 2024?",
        "How many course units are required for the Master of Language Technologies program?",
        "What is the application fee for the MSAII program?",
        "What are the core courses for MCDS students?",
        "When is the Spring Carnival Midway Opening Ceremony for CMU 2024 Spring Carnival?",
        "When is the CMU commencement ceremony?"
    ]
    retriever = BM25Retriever()
    # SageMakerLlama27B.set_up()

    for q in questions:
        a = SageMakerLlama27B.prompt_without_initialization(retriever, q, top_n=10, print_prompt=False, few_shot=True)
        print(f"Q: {q}")
        print(f"A: {a}")
        print("============================")
