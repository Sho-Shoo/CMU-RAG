# CMU Chatbot

## Introduction
This project presents a specialized question-answering chatbot designed to provide comprehensive insights into Carnegie Mellon University (CMU), with a focus on the Language Technologies Institute (LTI). It covers faculty research, course offerings, academic details, university events, and the historical narrative of CMU and its School of Computer Science (SCS). The chatbot utilizes the Retrieval-Augmented Generation (RAG) framework, marking a significant advancement in retrieval efficacy from traditional BM25-based methods to an embedding-based retrieval system. This enhancement notably improves recall, F1-score, and Exact Match metrics by [xxx]%, [xxx]%, and [xxx]%, respectively.

## Motivation
The motivation behind this project stems from the need to provide a one-stop solution for obtaining accurate and detailed information about CMU's LTI. It aims to bridge the gap between query and information retrieval through technological innovation, making information access seamless for students, faculty, and curious minds.

## Technologies Used
- Retrieval-Augmented Generation (RAG) framework
- Gemma and Llama large language models
- Embedding-based retrieval system

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/orangejustin/cmu-lti-qa-chatbot.git
```

Navigate to the project directory:

```bash
cd cmu-lti-qa-chatbot
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To start the chatbot, run the following script:

```bash
python chatbot.py
```

Follow the on-screen instructions to interact with the chatbot.

## Challenges and Improvements
During the project, challenges such as digit recognition were encountered, highlighting the need for data augmentation to enhance system precision. Future improvements could include the integration of more sophisticated data preprocessing methods and exploring additional language models for better performance.

## Acknowledgments
This project was made possible through the contributions of the CMU community, the open-source community, and the developers of the Gemma and Llama language models. Special thanks to everyone involved in providing insights, data, and support to make this chatbot a reality.

## License
This project is licensed under the [MIT License](LICENSE).
