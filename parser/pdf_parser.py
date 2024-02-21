import requests
import pdfplumber
import os

from io import StringIO

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

def download_pdf(url, save_path):
    # Download the PDF from the URL
    try:
        response = requests.get(url)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"PDF successfully downloaded and saved")
    except requests.RequestException as e:
        print(f"An error occurred while downloading the PDF: {e}")


# Split the doc into maxmimum length of 100 words
def split_doc(text, doc_max_len = 100):
    words = text.split()
    docs = []
    while len(words) > doc_max_len:
        docs.append(" ".join(words[:doc_max_len]) + "<sep>")
        words = words[doc_max_len:]
    docs.append(" ".join(words) + "<sep>")
    return "".join(docs)

# Using pdfplumber to extract text from PDF
def extract_text_from_pdf(pdf_path, pages_to_skip = 0, jump_footer = True):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Skip the table of contents
            if i < pages_to_skip: 
                continue
            
            # Extract the text from PDF
            current_page = page.extract_text()

            if jump_footer:
                # Remove the page footer
                footer_idx = current_page.rfind("\n")

                if footer_idx != -1:  
                    current_page = current_page[:footer_idx] 
            

            # Split this page into small doc
            current_page_split = split_doc(current_page)
            
            text += current_page_split

            # # Add a separator between pages
            # text += "<sep>"
    
    return text

# Using pdfminer to extract text from PDF
# Source: https://stackoverflow.com/questions/70170544/pdfplumber-extract-text-from-dynamic-column-layouts
def convert_pdf_to_string(file_path):
    output_string = StringIO()
    with open(file_path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        # Process each page in the PDF
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    return(output_string.getvalue())


if __name__ == "__main__":
    """
    Parse handbook PDF
    """

    handbook_urls = ["https://lti.cs.cmu.edu/sites/default/files/PhD_Student_Handbook_2023-2024.pdf",
                     "https://lti.cs.cmu.edu/sites/default/files/MLT%20Student%20Handbook%202023%20-%202024.pdf",
                     "https://lti.cs.cmu.edu/sites/default/files/MIIS%20Handbook_2023%20-%202024.pdf",
                     "https://lti.cs.cmu.edu/sites/default/files/MCDS%20Handbook%2023-24%20AY.pdf",
                     "https://msaii.cs.cmu.edu/sites/default/files/Handbook-MSAII-2022-2023.pdf"
                     ]
    
    num_content_pages = [8, 5, 5, 6, 5]
    
    for i, url in enumerate(handbook_urls):
        save_path = "raw_data/raw_pdf/" + url.split("/")[-1]

        # Download the PDF
        if not os.path.exists(save_path):
            download_pdf(url, save_path)
        else:
            print(f"{save_path} already exists.")

        text = extract_text_from_pdf(save_path, pages_to_skip=num_content_pages[i])

        file_path = "knowledge_source/" + url.replace(".pdf", ".txt").replace("/", "|")

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)
            print(f"Text has been successfully saved to {file_path}")

    """
    Parse CMU Fact sheet PDF
    """
            
    fact_sheet_url = "https://www.cmu.edu/about/cmu_fact_sheet_02.pdf"

    save_path = "raw_data/raw_pdf/" + fact_sheet_url.split("/")[-1]

    # Download the PDF
    if not os.path.exists(save_path):
        download_pdf(fact_sheet_url, save_path)
    else:
        print(f"{save_path} already exists.")

    ### Method 1: Using pdfplumber
    # text = extract_text_from_pdf(save_path, jump_footer=False)

    # file_path = "knowledge_source/" + fact_sheet_url.replace(".pdf", ".txt").replace("/", "|")

    # with open(file_path, 'w', encoding='utf-8') as file:
    #     file.write(text)
    #     print(f"Text has been successfully saved to {file_path}")
            
    ### Method 2: Using pdfminer
    file_path = 'raw_data/raw_pdf/cmu_fact_sheet_02.pdf' 

    # Using pdfminer to extract text from PDF and then manually clean it
    text = convert_pdf_to_string(file_path)

    file_path = "knowledge_source/" + fact_sheet_url.replace(".pdf", ".txt").replace("/", "|")

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)
        print(f"Text has been successfully saved to {file_path}")
