from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
# REPLACE WITH YOUR PATH
GOOGLE_DRIVE_PATH = 'todo'
GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/ANLP/ANLP_hw2/"
sys.path.append(GOOGLE_DRIVE_PATH)
from retriever.bm25_retriever import BM25Retriever
from retriever.base_retriever import BaseRetriever
from evaluation_metric.evaluation import f1_score, recall_score, exact_match_score

class Gemma7Bit:
    def __init__(self, retriever: BaseRetriever, top_n: int = 5):
        print("=================Loading Gemma 7B-it model...=================")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it")
        self.tokenizer =  AutoTokenizer.from_pretrained("google/gemma-7b-it", device_map="auto")
        self.retriever = retriever
        self.top_n = top_n

    def build_prompt(self, context: str, question: str, few_shot: bool = True):
        if few_shot:
            prompt_template = "You are a question-answering assistant who provides a short answer to a QUESTION "\
            "based on the CONTEXT about Carnegie Mellon University (CMU) and Language Technology Institute (LTI). "\
            "If the CONTEXT does not contain necessary information, please answer \"I don't know\". Please keep the answer short and simple. "\
            "Here are a few examples:\n\n"\
            "Question: When is 2024 Spring Carnival?\n"\
            "Answer: April 11 to April 14.\n\n"\
            "Question: When was Carnegie Mellon University founded?\n"\
            "Answer: Year 1900.\n\n"\
            "CONTEXT:{context}\n\n"\
            "QUESTION:{question}"


        else:
            prompt_template = "You are a question-answering assistant who provides a short answer to a QUESTION "\
            "based on the CONTEXT about Carnegie Mellon University (CMU) and Language Technology Institute (LTI). "\
            "If the CONTEXT does not contain necessary information, please answer \"I don't know\". Please keep the answer short and simple.\n\n"\
            "CONTEXT:{context}\n\n"\
            "QUESTION:{question}"
            

        input = prompt_template.replace("{context}", context).replace("{question}", question)

        chat = [{"role": "user", "content": input},
                {"role": "model", "content": "ANSWER:"}]
        
        input_with_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

        return input_with_prompt
    
    def answer(self, question: str, max_new_tokens: int = 512, top_k: float = 50, top_p: float = 0.9, temperature: float = 0.6, do_sample: bool = True, print_prompt=False):
        documents = self.retriever.retrieve(question, top_n=self.top_n)
        context = "\n".join(documents)
        input_with_prompt = self.build_prompt(context, question)

        if print_prompt: 
            print("****************Prompt****************")
            print(input_with_prompt)
            print("**************************************")

        inputs = self.tokenizer.encode(input_with_prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, top_p=top_p,top_k=top_k, temperature=temperature, do_sample=do_sample)
        # outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        answer = response.split("ANSWER:")[1].strip()
        return answer
    

if __name__ == "__main__":
    retriever = BM25Retriever()
    gemma = Gemma7Bit(retriever, top_n=10)

    questions = []
    ground_truths = []

    # Read the data
    with open(GOOGLE_DRIVE_PATH + "data/test/questions.txt", "r") as f:
        for q in f:
            questions.append(q.strip())

    with open(GOOGLE_DRIVE_PATH + "data/test/reference_answers.txt", "r") as f:
        for a in f:
            ground_truths.append([a.strip()])
    
    # Evaluate the model
    f1_scores = []
    recall_scores = []
    em_scores = []

    outputs = []

    for i, question in enumerate(questions):
        print(f"Question: {question}")
        print(f"Ground truth: {ground_truths[i]}")
        print("==========================================")
        print("Generating answer...")
        answer = gemma.answer(question, print_prompt=True)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Ref A: {ground_truths[i]}")
        print(f"Exact match score for the answer: {exact_match_score(answer, ground_truths[i])}")  
        print(f"F1 score for the answer: {f1_score(answer, ground_truths[i])}")
        print(f"Recall score for the answer: {recall_score(answer, ground_truths[i])}")
        
        # Store the answer
        outputs.append(answer)

        f1_scores.append(f1_score(answer, ground_truths[i]))
        recall_scores.append(recall_score(answer, ground_truths[i]))
        em_scores.append(exact_match_score(answer, ground_truths[i]))

        print("==========================================\n")

    print(f"Average exact match score: {sum(em_scores)/len(em_scores)}")
    print(f"Average F1 score: {sum(f1_scores)/len(f1_scores)}")
    print(f"Average recall score: {sum(recall_scores)/len(recall_scores)}")

    # Write the outputs to a txt file
    with open(GOOGLE_DRIVE_PATH + "data/test/system_outputs.txt", "w") as f:
        for output in outputs:
            f.write(output + "\n")
    