from bs4 import BeautifulSoup
from base_parser import BaseParser
import requests
import time


class LTIResearchPapersParser(BaseParser):
    def __init__(self, urls, year=2023):
        super().__init__(url='https://api.semanticscholar.org/graph/v1/paper/search')
        self.urls_faulty = urls
        self.year = year
        self.S2_API_KEY = 'scv8zP7sDUao0gvaUt1aN7iUttJdx4hwfjP0UtK0'
        self.result_limit = 300

    def scrape_faculty_names(self):
        all_faculty_names = []
        for url in self.urls_faulty:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            faculty_names = [name_tag.text.strip() for name_tag in soup.select('.views-field.views-field-nothing h2')]
            all_faculty_names.extend(faculty_names)
        return all_faculty_names

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
            params={'fields': 'title,abstract,authors,citationCount,venue,year,tldr'},
            json={"ids": paper_ids}

        )
        response.raise_for_status()

        try:
            response.raise_for_status()
            paper_details = response.json()
        except Exception:
            print('Could not fetch tldr')
            return 'The paper did have tldr'
        return paper_details

    def parse(self):
        faculty_names = self.scrape_faculty_names()
        for name in faculty_names:
            print(f"Searching for: {name}")
            author_id = self.find_author_id_by_name(name)
            if author_id:
                print(f"Found author ID {author_id} for {name}. Fetching papers...")
                papers = self.fetch_papers_for_author(author_id)
                outputs = self.fetch_paper_details_with_tldr([paper['paperId'] for paper in papers])
                time.sleep(2)
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
                    for output in outputs
                ])
                self._write_doc(doc)
                self._save_file()
            else:
                print(f"No author ID found for {name}.")



if __name__ == '__main__':
    # Fetch faculty's name in LTI
    lti_urls = ['https://lti.cs.cmu.edu/directory/all/154/1',
                'https://lti.cs.cmu.edu/directory/all/154/1?page=1']
    parser = LTIResearchPapersParser(urls=lti_urls, year=2023)
    parser.parse()
