## Running the RAG system

- Set up a Conda environment by running `./setup.sh`
- Install dependencies using `pip install -r requirements.txt`
- Run either `language_model/llama2_7b.py` or `language_model/gemma_7b_it.py` as they both contain a short script that reads questions from a .txt file, answers them, and writes answers to another .txt file.
- You can also swap the retrievers in both scripts by trying out either `BM25Retriever` or `EmbeddingRetriever`.

**Note that a complete setup might requirement AWS access as both scripts require SageMaker endpoints to run inference on the LLMs.**

## Repository structure

```commandline
├── LICENSE
├── README.md
├── contributions.md
├── data
│   ├── test                                     # evaluation data QA pairs and question category information
│   │   ├── question_categories.txt
│   │   ├── questions.txt
│   │   └── reference_answers.txt
│   └── train                                    # in-context learning examples
│       ├── questions.txt
│       └── reference_answers.txt
├── evaluation_metric                                  
│   └── evaluation.py                            # F1, recall, exact match metric calculation
├── faculty_info.csv
├── knowledge_source                             # plain-text-stored knowledge source
├── knowledge_source_pd                          # tabular stored knowledge source
├── language_model
│   ├── aws_config.py
│   ├── gemma_7b_it.py                           # Gemma 7B prompting script
│   ├── llama2_7b.py                             # Llama 7B prompting script
│   └── utils.py
├── parser                                       # parsers for various data sources
│   ├── api_parser.py
│   ├── api_parser_pd.py
│   ├── base_parser.py
│   ├── excel_parser.py
│   ├── faculty_parser.py
│   ├── html_parser.py
│   ├── pdf_parser.py
│   ├── schedule_parser.py
│   └── schedule_parser_pd.py
├── prompt_template
│   ├── version1
│   │   ├── gemma_prompt_v1.py
│   │   └── prompt_v1.md
│   └── version2
│       ├── gemma_prompt_v2.py
│       └── prompt_v2.md
├── requirements.txt
├── retriever                                    # implementation of BM25 and ChromaDB-based embedding retriever
│   ├── base_retriever.py
│   ├── bm25_retriever.py
│   ├── chroma_meta                              # ChromaDB storage
│   ├── embedding_retriever.py
│   └── Embedding_Retriever.ipynb
├── setup.sh
└── system_outputs                               # system output of final submission
    ├── system_output_1.txt
    ├── system_output_2.txt
    └── system_output_3.txt
```