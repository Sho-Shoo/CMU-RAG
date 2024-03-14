_IN_COMTEXT_EXAMPLES = \
"""
Q: Where is Professor Graham Neubig's office?
A: Professor Graham Neubig's office is Room 5409 of the Gates & Hillman Centers.

Q: Who is teaching 11711 in Fall 2023?
A: Professor Robert Frederking and Daniel Fried.

Q: Who are the co-authors of the paper "Generating Images with Multimodal Language Models" published in Neural Information Processing Systems in 2023, and what is the main proposal of this research?
A: The co-authors of the paper are Jing Yu Koh, Daniel Fried, and R. Salakhutdinov. The main proposal of this research is a method to fuse frozen text-only large language models (LLMs) with pre-trained image encoder and decoder models, by mapping between their embedding spaces, and exhibits a wider range of capabilities compared to prior multimodal language models.

Q: What is Chute Flagger?
A: Chute Flagger is team member who provides a signal for buggy drivers to know when to start the right-hand turn from Schenley Drive onto Frew Street.
"""


def get_in_context_example() -> str:
    return _IN_COMTEXT_EXAMPLES
