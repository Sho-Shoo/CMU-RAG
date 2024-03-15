import pandas as pd
from base_parser import BaseParser
import requests
from bs4 import BeautifulSoup, NavigableString

def _extract_contact(soup, tag, class_name):
    container = soup.find(tag, class_=class_name)
    if container:
        text = container.text.strip()
        return text
    return None

def _extract_research_area(soup):
    container = soup.find('h2', string=lambda x: x == 'Research Area')
    if container:
        text = container.find_next_sibling('p').get_text(strip=True)
        return text
    return None

def _extract_research(soup):
    container = soup.find('h2', string=lambda x: x == 'Research')
    if container:
        text = ""
        for sibling in container.find_next_siblings():
            if sibling.name == 'h2':
                break
            if not isinstance(sibling, NavigableString):
                text += sibling.get_text(separator="\n", strip=True)
        return text
    return None

def _extract_other_context(soup, heading):
    container = soup.find('h2', string=lambda x: x == heading)
    text = ""
    if container:
        for sibling in container.find_next_siblings():
            if sibling.name == 'h2':
                break
            if sibling.name == 'p':
                text += sibling.get_text(separator="\n", strip=True)
        return text
    return None

def _extract_education(soup):
    container = soup.find('h2', string=lambda x: x == 'Education')
    if container:
        context = container.find_next_sibling('span')
        if context == None:
            text = container.find_next_sibling(string=True).strip()
        else:
            text = context.get_text(strip=True)
        return text
    return None


class FacultyInfoParser(BaseParser):
    # Override
    def parse(self) -> None:
        response = requests.get(self.url)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract name
        name = soup.find('title').text.split(' - ')[0].strip()

        # Extract title
        title = soup.find('h2', style="font-size:1.15em").text.strip()

        # Extract contact information
        office = _extract_contact(soup, 'span', 'icon loc')
        if office: office = office.replace("—", "")
        email = _extract_contact(soup, 'span', 'protect hidden')
        if email: email = email.replace('(through)', '@')
        phone = _extract_contact(soup, 'a', 'icon tel')

        # Extract research information
        research_area = _extract_research_area(soup)
        research = _extract_research(soup)

        # Extract projects
        projects = _extract_other_context(soup, 'Projects')

        # Extract Bio
        bio = _extract_other_context(soup, 'Bio')

        # Extract education
        education = _extract_education(soup)

        # Write to file
        info = {
            f"Professor {name} Title": title,
            f"Professor {name} Office": office,
            f"Professor {name} Email": email,
            f"Professor {name} Phone": phone,
            f"Professor {name} Research Area": research_area,
            f"Professor {name} Research": research,
            f"Professor {name} Projects": projects,
            f"Professor {name} Bio": bio,
            f"Professor {name} Education": education
        }

        kv_texts = []

        for key, value in info.items():
            kv_text = f"{key}: {value}"
            kv_texts.append(kv_text)
        entry_text = "<sep>".join(kv_texts)
        
        self._write_doc(entry_text)

        self._save_file()

        # Convert to dataframe
        info_df = pd.DataFrame([info])

        return info_df


if __name__ == "__main__":
    faculty_info_url = ["https://lti.cs.cmu.edu/people/faculty/bisk-yonatan.html",
                        "https://lti.cs.cmu.edu/people/faculty/brown-ralf.html",
                        "https://lti.cs.cmu.edu/people/faculty/callan-jamie.html",
                        "https://lti.cs.cmu.edu/people/faculty/cassell-justine.html",
                        "https://lti.cs.cmu.edu/people/faculty/diab-mona.html",
                        "https://lti.cs.cmu.edu/people/faculty/diaz-fernando.html",
                        "https://lti.cs.cmu.edu/people/faculty/fahlman-scott.html",
                        "https://lti.cs.cmu.edu/people/faculty/frederking-robert.html",
                        "https://lti.cs.cmu.edu/people/faculty/fried-daniel.html",
                        "https://lti.cs.cmu.edu/people/faculty/hauptmann-alexander.html",
                        "https://lti.cs.cmu.edu/people/faculty/ippolito-daphne.html",
                        "https://lti.cs.cmu.edu/people/faculty/levin-lori.html",
                        "https://lti.cs.cmu.edu/people/faculty/bio.html",
                        "https://lti.cs.cmu.edu/people/faculty/mitamura-teruko.html",
                        "https://lti.cs.cmu.edu/people/faculty/morency-louis-philippe.html",
                        "https://lti.cs.cmu.edu/people/faculty/mortensen-david.html",
                        "https://lti.cs.cmu.edu/people/faculty/neubig-graham.html",
                        "https://lti.cs.cmu.edu/people/faculty/nyberg-eric.html",
                        "https://lti.cs.cmu.edu/people/faculty/oflazer-kemal.html",
                        "https://lti.cs.cmu.edu/people/faculty/raj-bhiksha.html",
                        "https://lti.cs.cmu.edu/people/faculty/ros%C3%A9-carolyn.html",
                        "https://lti.cs.cmu.edu/people/faculty/rudnicky-alexander.html",
                        "https://lti.cs.cmu.edu/people/faculty/sap-maarten.html",
                        "https://lti.cs.cmu.edu/people/faculty/shamos-michael.html",
                        "https://lti.cs.cmu.edu/people/faculty/singh-rita.html",
                        "https://lti.cs.cmu.edu/people/faculty/strubell-emma.html",
                        "https://lti.cs.cmu.edu/people/faculty/waibel-alexander.html",
                        "https://lti.cs.cmu.edu/people/faculty/watanabe-shinji.html",
                        "https://lti.cs.cmu.edu/people/faculty/welleck-sean.html",
                        "https://lti.cs.cmu.edu/people/faculty/xing-eric.html",
                        "https://lti.cs.cmu.edu/people/faculty/xiong-chenyan.html",
                        "https://lti.cs.cmu.edu/people/faculty/yiming-yang.html"]

    save_csv = False
    info_dfs = []
    for url in faculty_info_url:
        parser = FacultyInfoParser(url, doc_max_len=500)
        info_df = parser.parse()
        info_dfs.append(info_df)
    
    if save_csv:
        all_info_df = pd.concat(info_dfs, ignore_index=True)
        all_info_df.to_csv("knowledge_source_pd/faculty_info.csv", index=False)
