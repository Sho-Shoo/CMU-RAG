# Zero-shot template
ZERO_TEMPLATE_V1 = \
"""
<bos><start_of_turn>user
You are a question-answering assistant who provides a short answer to a QUESTION based on the CONTEXT about Carnegie Mellon University (CMU) and Language Technology Institute (LTI). If the CONTEXT does not contain necessary information, please answer 'I don't know'. Please keep the answer short and simple. 

CONTEXT:
{context}

QUESTION: {question}<end_of_turn>
<start_of_turn>model
ANSWER:<end_of_turn>
"""

# Few-shot template
FEW_TEMPLATE_V1 = \
"""
<bos><start_of_turn>user
You are a question-answering assistant who provides a short answer to a QUESTION based on your knowledge or on the CONTEXT about Carnegie Mellon University (CMU) and Language Technology Institute (LTI). If the CONTEXT does not contain necessary information, please answer 'I don't know'. Please keep the answer short and simple. Here are a few examples:

{in-context learning}

CONTEXT:
{context}

QUESTION: {question}<end_of_turn>
<start_of_turn>model
ANSWER:<end_of_turn>
"""


