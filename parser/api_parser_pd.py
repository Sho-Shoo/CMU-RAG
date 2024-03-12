from api_parser import LTIResearchPapersParser
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time

def scrape_faculty_names():
    response = requests.get('https://lti.cs.cmu.edu/people/faculty/index.html')
    soup = BeautifulSoup(response.content, 'html.parser')
    faculty_names = soup.find_all('a', class_='name')
    names = [tag.text.strip() for tag in faculty_names]
    names = [' '.join(name.split()) for name in names]
    return names

class PaperParserToDataframe(LTIResearchPapersParser):
    def __init__(self, year, output_dir, category):
        super().__init__(year)
        self.output_dir = output_dir
        self.category = category

    # Overrides parse
    def parse(self):
        faculty_names = scrape_faculty_names()
        papers_df = pd.DataFrame(
            columns=['Author', 'Title', 'Authors', 'Abstract', 'Year', 'Venue', 'Citations', 'TLDR'])
        all_paper_data = []

        for name in faculty_names:
            print(f"Searching for: {name}")
            author_id = self.find_author_id_by_name(name)
            if author_id:
                if author_id:
                    if author_id == '2256564956':
                        author_id = '1783635'
                    if author_id == '17038253':
                        author_id = '144987107'
                    if author_id == '2107366678':
                        author_id = '1746678'
                    if author_id == '2109454364':
                        author_id = '153915824'
                    if author_id == '2082457533':
                        author_id = '3407646'
                    if author_id == '11908065':
                        author_id = '49933077'
                    if author_id == '9273426':
                        author_id = '35729970'
                    if author_id == '2064429921':
                        author_id = '1724972'
                print(f"Found author ID {author_id} for {name}. Fetching papers...")
                papers = self.fetch_papers_for_author(author_id)
                papers = [item['paperId'] for item in papers if item['year'] == 2023]
                paper_chunks = [papers[i:i + 300] for i in range(0, len(papers), 300)]
                for chunk in paper_chunks:
                    outputs = self.fetch_paper_details_with_tldr(chunk)

                    if outputs == 'The paper did not have tldr':
                        continue

                    time.sleep(1)

                    for output in outputs:
                        paper_data = {
                            'Author': name,
                            'Title': output['title'],
                            'Authors': ', '.join([author['name'] for author in output['authors']]),
                            'Abstract': output['abstract'],
                            'Year': output['year'],
                            'Venue': output['venue'],
                            'Citations': output['citationCount'],
                            'TLDR': output['tldr']
                        }
                        all_paper_data.append(paper_data)
            else:
                print(f"No author ID found for {name}.")

        papers_df = pd.DataFrame(all_paper_data)
        # save the DataFrame to a CSV file
        papers_df.to_csv(os.path.join(self.output_dir, 'papers.csv'), index=False)


if __name__ == '__main__':
    parser = PaperParserToDataframe(year=2023,
                                    output_dir='knowledge_source_pd',
                                    category='papers')
    parser.parse()