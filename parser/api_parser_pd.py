from api_parser import LTIResearchPapersParser
import pandas as pd
import os
import time

class PaperParserToDataframe(LTIResearchPapersParser):
    def __init__(self, year, output_dir, category):
        super().__init__(year)
        self.output_dir = output_dir
        self.category = category

    # Overrides parse
    def parse(self):
        name = pd.read_csv(os.path.join('faculty_info.csv'))
        name = name.set_index('author_name').to_dict('index')
        papers_df = pd.DataFrame(
            columns=['Author', 'Title', 'Authors', 'Abstract', 'Year', 'Venue', 'Citations', 'TLDR'])
        all_paper_data = []

        for name, id_to_num in name.items():
            print(f"Searching for: {name}")
            author_id = id_to_num['author_id']
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

        papers_df = pd.DataFrame(all_paper_data)
        # save the DataFrame to a CSV file
        papers_df.to_csv(os.path.join(self.output_dir, 'papers/papers.csv'), index=False)


if __name__ == '__main__':
    parser = PaperParserToDataframe(year=2023,
                                    output_dir='knowledge_source_pd',
                                    category='papers')
    parser.parse()