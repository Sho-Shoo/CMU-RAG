from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
# REPLACE WITH YOUR PATH
GOOGLE_DRIVE_PATH = 'todo'
sys.path.append(GOOGLE_DRIVE_PATH)
from retriever.bm25_retriever import BM25Retriever
from retriever.base_retriever import BaseRetriever
from evaluation_metric.evaluation import f1_score

class Gemma7Bit:
    def __init__(self, retriever: BaseRetriever, top_n: int = 5):
        print("=================Loading Gemma 7B-it model...=================")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it")
        self.tokenizer =  AutoTokenizer.from_pretrained("google/gemma-7b-it", device_map="auto")
        self.retriever = retriever
        self.top_n = top_n

    def build_prompt(self, context: str, question: str):
        prompt_template = "You are a helpful question-answering assistant who focuses specifically on questions"\
        "about various facts about Carnegie Mellon University (CMU) and Language Technologies Institute (LTI)."\
        "Please answer the following QUESTION based on this following step:\n"\
        "1. If you know the answer based on your knowledge, directly answer.\n"\
        "2. If you are not sure about the answer, you could refer to the CONTEXT to find the answer.\n"\
        "3. If you do not know the answer and the CONTEXT does not contain the answer, truthfully say, \"I don't know\".\n"\
        "Please keep the answer simple and straight-forward.\n\n"\
        "CONTEXT:\n{context}\n\n"\
        "QUESTION:\n{question}"

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
        # outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, top_p=top_p,top_k=top_k, temperature=temperature, do_sample=do_sample)
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        answer = response.split("ANSWER:")[1].strip()
        return answer
    

if __name__ == "__main__":
    retriever = BM25Retriever()
    gemma = Gemma7Bit(retriever, top_n=10)

    questions = [
        "When was Carnegie Mellon University founded?",
        "Who is the president of CMU?",
        "What are the research interests of Graham Neubig?",
        "What is the mascot of CMU?",
        "What courses are offered by Graham Neubig at CMU?",
        "Who teaches 11711 Advanced Natural Language Processing in Spring 2024?"
    ]
    ground_truths = [["Carnegie Mellon University was founded in 1900."], 
                     ["Farnam Jahanian is the president of CMU."], 
                     ["Graham Neubig's research interests include Machine Translation, Natural Language Processing, and Machine Learning."], 
                     ["The mascot of CMU is Scottish terrier."], 
                     ["Graham Neubig offers 11711 Advanced Natural Language Processing in Spring 2024."], 
                     ["Graham Neubig teaches 11711 Advanced Natural Language Processing in Spring 2024."]]

    for i, question in enumerate(questions):
        print("==========================================")
        print("Generating answer...")
        answer = gemma.answer(question, print_prompt=True)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Ref A:{ground_truths[i]}")
        print(f"F1 score for the answer: {f1_score(answer, ground_truths[i])}")
        print("==========================================\n")