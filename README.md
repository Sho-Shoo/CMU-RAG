## Running the RAG system

- Set up a Conda environment by running `./setup.sh`
- Install dependencies using `pip install -r requirements.txt`
- Run either `language_model/llama2_7b.py` or `language_model/gemma_7b_it.py` as they both contain a short script that reads questions from a .txt file, answers them, and writes answers to another .txt file.
- You can also swap the retrievers in both scripts by trying out either `BM25Retriever` or `EmbeddingRetriever`.

**Note that a complete setup might requirement AWS access as both scripts require SageMaker endpoints to run inference on the LLMs.**