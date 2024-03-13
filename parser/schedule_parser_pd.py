from schedule_parser import ScheduleParser
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re

class ScheduleParserToDataframe(ScheduleParser):

    def __init__(self, url, schedule_title, output_dir):
        super().__init__(url, schedule_title)
        self.output_dir = output_dir

    # Override
    def parse(self) -> None:
        response = requests.get(self.url)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')
        rows = table.find_all('tr')

        # Initialize a list to store each entry's dictionary
        schedule_data = []

        for row in rows:
            cells = row.find_all('td')
            if len(cells) != 10:
                continue

            if all(cell.text.strip() != "\xa0" for cell in cells[:3]):
                if cells[0].text.strip() == '':
                    continue
                entry = {
                    "schedule_title": self.schedule_title,
                    "course_number": cells[0].text.strip(),
                    "title": cells[1].text.strip(),
                    "units": cells[2].text.strip(),
                    "section": cells[3].text.strip(),
                    "day": cells[4].text.strip(),
                    "begin": cells[5].text.strip(),
                    "end": cells[6].text.strip(),
                    "room": cells[7].text.strip(),
                    "location": cells[8].text.strip(),
                    "instructor": cells[9].text.strip()
                }
                schedule_data.append(entry)

        schedule_df = pd.DataFrame(schedule_data)

        # save the DataFrame to a CSV file
        csv_filename = f"{self.schedule_title.replace(' ', '_').lower()}_schedule.csv"
        schedule_df.to_csv(os.path.join(self.output_dir, csv_filename), index=False)
        print(f"Schedule saved to {csv_filename}")


if __name__ == "__main__":
    urls = ['https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_summer_2.htm',
             'https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_spring.htm',
             'https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_fall.htm',
             'https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_summer_1.htm']
    keys = ['Summer Two 2024', 'Spring 2024', 'Fall 2023', 'Summer One 2024']

    for key, url in zip(keys, urls):
        parser = ScheduleParserToDataframe(url=url,
                                           schedule_title=key,
                                           output_dir='knowledge_source_pd/courses'
                                           )
        parser.parse()
