import sys
path = "/Users/a119491/Library/Mobile Documents/com~apple~CloudDocs/11711/ASSGN/ASSGN2/ANLP-HW2/"
sys.path.append(path)
from retriever.bm25_retriever import BM25Retriever
from retriever.base_retriever import BaseRetriever
from retriever.embedding_retriever import EmbeddingRetriever
from evaluation_metric.evaluation import f1_score, recall_score, exact_match_score, write_test_result
import boto3
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel
import json
from aws_config import AWSConfig
from pprint import pp, pformat
from prompt_template.version1.gemma_prompt_v1 import ZERO_TEMPLATE_V1, FEW_TEMPLATE_V1
from prompt_template.version2.gemma_prompt_v2 import ZERO_TEMPLATE_V2, FEW_TEMPLATE_V2
from language_model.utils import get_in_context_example

# Helper function to build the prompt
def _build_gemma_prompt(context: str, question: str, few_shot: bool = True, template_ver = 1) -> str:
    if template_ver == 1:
        if few_shot:
            template = FEW_TEMPLATE_V1
            template = template.replace("{in-context learning}", get_in_context_example())
        else:
            template = ZERO_TEMPLATE_V1
    else:
        if few_shot:
            template = FEW_TEMPLATE_V2
            template = template.replace("{in-context learning}", get_in_context_example())
        else:
            template = ZERO_TEMPLATE_V2
    
    prompt = template.replace("{context}", context).replace("{question}", question)

    return prompt


class SageMakerGemma7Bit:
    @classmethod
    def set_up(cls):
        role = AWSConfig.SAGEMAKER_ARN_ROLE
        gemma_model = JumpStartModel(model_id="huggingface-llm-gemma-7b-instruct", role=role)
        print("Deploying Gemma 7b-it to SageMaker...")
        gemma_model.deploy(initial_instance_count=1,
                           endpoint_name=AWSConfig.SAGEMAKER_ENDPOINT_NAME,
                           accept_eula=True)

    @classmethod
    def shut_down(cls):
        print(f"Shutting down Sagemaker endpoint {AWSConfig.SAGEMAKER_ENDPOINT_NAME}...")
        sagemaker_session = sagemaker.Session()
        sagemaker_session.delete_endpoint(AWSConfig.SAGEMAKER_ENDPOINT_NAME)
        sagemaker_session.delete_endpoint_config(AWSConfig.SAGEMAKER_ENDPOINT_NAME)

    @classmethod
    def generate(cls, retriever: BaseRetriever, question: str, top_n: int = 10,
                max_new_tokens: int = 1024,  top_k: float = 50, top_p: float = 0.9, 
                temperature: float = 0.7, do_sample: bool = True,
                print_prompt=False, few_shot=True, template_ver=1):
        
        # Retrieve the documents
        documents = retriever.retrieve(question, top_n=top_n)
        context = "\n".join(documents)

        # Build the prompt
        prompt = _build_gemma_prompt(context, question, few_shot=few_shot, template_ver=template_ver)
        if print_prompt: 
            print(prompt)

        # Send the input to the model
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_new_tokens,
                           "top_k": top_k,
                           "top_p": top_p,
                           "temperature": temperature,
                            "do_sample": do_sample,
                           "return_full_text": False}
        }

        runtime = boto3.client("runtime.sagemaker")
        payload = json.dumps(payload, indent=4).encode('utf-8')
        response = runtime.invoke_endpoint(EndpointName=AWSConfig.SAGEMAKER_ENDPOINT_NAME,
                                           ContentType="application/json",
                                           Body=payload)
        
        return json.loads(response["Body"].read())[0]["generated_text"]
    

if __name__ == "__main__":
    USE_BM25 = False
    FEW_SHOT = False
    TOP_N = 10 if USE_BM25 else 3
    TEM_VER = 2

    file_name_map = {
        (False, False, 1): "prompt_v1/gemma_embed_zero.txt",
        (False, True, 1): "prompt_v1/gemma_embed_few.txt",
        (False, False, 2): "prompt_v2/gemma_embed_zero.txt",
        (False, True, 2): "prompt_v2/gemma_embed_few.txt",
        (True, False, 1): "prompt_v1/gemma_bm25_zero.txt",
        (True, True, 1): "prompt_v1/gemma_bm25_few.txt",
        (True, False, 2): "prompt_v2/gemma_bm25_zero.txt",
        (True, True, 2): "prompt_v2/gemma_bm25_few.txt"
    }

    result_file_name = file_name_map[(USE_BM25, FEW_SHOT, TEM_VER)]

    # Read the data
    questions = []
    ground_truths = []
    question_category = []

    with open("data/test/questions.txt", "r") as f:
        for q in f:
            questions.append(q.strip())

    with open("data/test/reference_answers.txt", "r") as f:
        for a in f:
            ground_truths.append([a.strip()])

    with open("data/test/question_categories.txt", "r") as f:
        for c in f:
            question_category.append(c.strip())
    
    # Evaluate the model
    f1_scores = []
    recall_scores = []
    em_scores = []
    cate_summary = dict()

    outputs = []

    # Set up retriever
    if USE_BM25:
        retriever = BM25Retriever()
    else:
        retriever = EmbeddingRetriever(TOP_N)

    # Set up Gemma model
    gemma = SageMakerGemma7Bit()
    # gemma.set_up()

    for i, question in enumerate(questions):
        print("==========================================")
        print(f"Q: {question}")
        print("Generating answer...")
        answer = gemma.generate(retriever, question, top_n=TOP_N, print_prompt=True, few_shot=FEW_SHOT, template_ver=TEM_VER)
        print(f"A: {answer}")
        print(f"Ref A: {ground_truths[i]}")
        print(f"Exact match score for the answer: {exact_match_score(answer, ground_truths[i])}")  
        print(f"F1 score for the answer: {f1_score(answer, ground_truths[i])}")
        print(f"Recall score for the answer: {recall_score(answer, ground_truths[i])}")
        
        # Store the answer
        outputs.append(answer)
        
        f1 = f1_score(answer, ground_truths[i])
        recall = recall_score(answer, ground_truths[i])
        em = exact_match_score(answer, ground_truths[i])

        f1_scores.append(f1)
        recall_scores.append(recall)
        em_scores.append(em)

        # Store the category summary
        category = question_category[i]
        if category not in cate_summary:
            cate_summary[category] = {
                "f1": [],
                "recall": [],
                "em": []
            }

        cate_summary[category]["f1"].append(f1)
        cate_summary[category]["recall"].append(recall)
        cate_summary[category]["em"].append(em)
        print("==========================================\n")


    # Calculate the average scores
    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)

    for category in cate_summary:
        for metric in ["f1", "recall", "em"]:
            scores = cate_summary[category][metric]
            cate_summary[category][metric] = sum(scores) / len(scores)

    test_summary = f"Average F1 score: {avg_f1}\n" + \
                   f"Average Recall score: {avg_recall}\n" + \
                   f"Average EM score: {avg_em}\n" + \
                   pformat(cate_summary)
    
    print(test_summary)

    # Write the outputs to a txt file
    write_test_result("data/test/" + result_file_name, outputs, test_summary)

    #gemma.shut_down()