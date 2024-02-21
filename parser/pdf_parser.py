import requests
import pdfplumber
import os

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


def extract_text_from_pdf(pdf_path, pages_to_skip = 0):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if i < pages_to_skip:  # Skip pages that are likely part of the TOC
                continue
            # Extract the text from PDF
            text += page.extract_text()

            # Remove the page footer
            footer_idx = text.rfind("\n")
            if footer_idx != -1:  
                text = text[:footer_idx] 
            
            # Add a separator between pages
            text += "<sep>"
    
    return text


if __name__ == "__main__":
    handbook_urls = ["https://lti.cs.cmu.edu/sites/default/files/PhD_Student_Handbook_2023-2024.pdf",
                     "https://lti.cs.cmu.edu/sites/default/files/MLT%20Student%20Handbook%202023%20-%202024.pdf",
                     "https://lti.cs.cmu.edu/sites/default/files/MIIS%20Handbook_2023%20-%202024.pdf",
                     "https://lti.cs.cmu.edu/sites/default/files/MCDS%20Handbook%2023-24%20AY.pdf",
                     "https://msaii.cs.cmu.edu/sites/default/files/Handbook-MSAII-2022-2023.pdf"
                     ]
    num_content_pages = [8, 5, 5, 6, 5]
    
    for i, url in enumerate(handbook_urls):
        save_path = "raw_pdf/" + url.split("/")[-1]

        if not os.path.exists(save_path):
            download_pdf(url, save_path)
        else:
            print(f"{save_path} already exists.")

        text = extract_text_from_pdf(save_path, pages_to_skip=num_content_pages[i])

        file_path = "knowledge_source/" + url.replace(".pdf", ".txt").replace("/", "|")

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)
            print(f"Text has been successfully saved to {file_path}")

