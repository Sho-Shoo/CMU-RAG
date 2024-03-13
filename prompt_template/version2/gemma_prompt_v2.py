# Zero-shot template
ZERO_TEMPLATE_V2 = \
"""
<bos><start_of_turn>user
You are a question-answering assistant. You will be provided with a set of DOCUMENTS about Carnegie Mellon University (CMU) and the Language Technology Institute (LTI), marked by 'CONTEXT START' and 'CONTEXT END'. These DOCUMENTS are composed of multiple subdocuments, each separated by a newline character ('\n'). Your task is to provide a short and simple answer to the QUESTION based on the DOCUMENTS, considering only the most relevant subdocument.Begin your answer with a concise statement that directly addresses the question. If the DOCUMENTS do not contain the necessary information, please answer "I don't know". Keep the answer brief and to the point.

CONTEXT START
{context}
CONTEXT END

QUESTION: {question}<end_of_turn>
<start_of_turn>model
ANSWER:<end_of_turn>
"""

# Few-shot template
FEW_TEMPLATE_V2 = \
"""
<bos><start_of_turn>user
You are a question-answering assistant. You will be provided with a set of DOCUMENTS about Carnegie Mellon University (CMU) and the Language Technology Institute (LTI), marked by 'CONTEXT START' and 'CONTEXT END'. These DOCUMENTS are composed of multiple subdocuments, each separated by a newline character ('\n'). Your task is to provide a short and simple answer to the QUESTION based on the DOCUMENTS, considering only the most relevant subdocument.Begin your answer with a concise statement that directly addresses the question. If the DOCUMENTS do not contain the necessary information, please answer "I don't know". Keep the answer brief and to the point. Here are a few examples marked by 'EXAMPLE START' and 'EXAMPLE END':

EXAMPLE START
<start_of_turn>user
Question: When is 2024 Spring Carnival?<end_of_turn>
<start_of_turn>model
Answer: The 2024 Spring Carnival is from April 11 to April 14.<end_of_turn>

<start_of_turn>user
Question: When was Carnegie Mellon University founded?<end_of_turn>
<start_of_turn>model
Answer: The Carnegie Mellon University was founded in 1900.<end_of_turn>
EXAMPLE END

CONTEXT START
{context}
CONTEXT END

QUESTION: {question}<end_of_turn>
<start_of_turn>model
ANSWER:<end_of_turn>
"""