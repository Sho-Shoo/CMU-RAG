from bs4 import BeautifulSoup
from base_parser import BaseParser
import requests
import time


def scrape_faculty_names():
    response = requests.get('https://lti.cs.cmu.edu/people/faculty/index.html')
    soup = BeautifulSoup(response.content, 'html.parser')
    faculty_names = soup.find_all('a', class_='name')
    names = [tag.text.strip() for tag in faculty_names]
    names = [' '.join(name.split()) for name in names]
    return names

class LTIResearchPapersParser(BaseParser):
    def __init__(self, year=2023):
        super().__init__(url='https://api.semanticscholar.org/graph/v1/paper/search')
        self.year = year
        self.S2_API_KEY = 'scv8zP7sDUao0gvaUt1aN7iUttJdx4hwfjP0UtK0'
        self.result_limit = 1000

    def find_author_id_by_name(self, name):
        search_url = 'https://api.semanticscholar.org/graph/v1/author/search'
        params = {'query': name, 'limit': 1, 'fields': 'authorId'}
        headers = {'x-api-key': self.S2_API_KEY}

        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        authors = data.get('data', [])

        if authors:
            return authors[0].get('authorId')
        return None

    def fetch_papers_for_author(self, author_id):
        if author_id:
            papers_url = f'https://api.semanticscholar.org/graph/v1/author/{author_id}/papers'
            params = {'limit': self.result_limit, 'fields': ''}
            headers = {'x-api-key': self.S2_API_KEY}

            response = requests.get(papers_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            return data.get('data', [])
        return []

    def fetch_paper_details_with_tldr(self, paper_ids):
        # Make the POST request to fetch details
        response = requests.post(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            headers={'x-api-key': self.S2_API_KEY},
            params={'limit': self.result_limit, 'fields': 'title,abstract,authors,citationCount,venue,year,tldr'},
            json={"ids": paper_ids}
        )

        try:
            response.raise_for_status()
            paper_details = response.json()
        except Exception:
            print('Could not fetch tldr')
            return 'The paper did not have tldr'
        return paper_details

    def parse(self):
        faculty_names = scrape_faculty_names()
        for name in faculty_names:
            print(f"Searching for: {name}")
            author_id = self.find_author_id_by_name(name)
            if author_id:
                print(f"Found author ID {author_id} for {name}. Fetching papers...")
                papers = self.fetch_papers_for_author(author_id)
                paper_chunks = [papers[i:i + 300] for i in range(0, len(papers), 300)]
                for chunk in paper_chunks:
                    outputs = self.fetch_paper_details_with_tldr([paper['paperId'] for paper in chunk])
                    if outputs == 'The paper did not have tldr':
                        print(outputs)
                        continue
                    time.sleep(1)
                    sep = '; '
                    doc = "\n".join([
                        f"Author (LTI's Professor): {name}{sep}"
                        f"Title: {output['title']}{sep}"
                        f"Authors: {', '.join([author['name'] for author in output['authors']])}{sep}"
                        f"Abstract: {output['abstract']}{sep}"
                        f"Year: {output['year']}{sep}"
                        f"Venue: {output['venue']}{sep}"
                        f"Citations: {output['citationCount']}{sep}"
                        f"TLDR: {output['tldr']}"
                        for output in outputs if output is not None  # Assuming None is returned for papers without TLDR
                    ])
                    self._write_doc(doc)
                    self._save_file()
            else:
                print(f"No author ID found for {name}.")

if __name__ == '__main__':
    parser = LTIResearchPapersParser(year=2023)
    parser.parse()
