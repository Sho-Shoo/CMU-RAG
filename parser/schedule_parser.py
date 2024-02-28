from parser.base_parser import BaseParser
import requests
from bs4 import BeautifulSoup


class ScheduleParser(BaseParser):

    def __init__(self, url: str, schedule_title: str):
        super().__init__(url)
        self.schedule_title = schedule_title

    def parse(self) -> None:
        response = requests.get(self.url)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')
        rows = table.find_all('tr')

        course_id, title, units = "", "", ""
        for row in rows:  # for each table row
            cells = row.find_all('td')  # get all table data

            if cells and cells[0] and cells[0].text != "\xa0":
                course_id = cells[0].text
            if cells and len(cells) > 1 and cells[1] and cells[1].text != "\xa0":
                title = cells[1].text
            if cells and len(cells) > 2 and cells[2] and cells[2].text != "\xa0":
                units = cells[2].text

            # if some fields are missing, skip
            if len(cells) != 10:
                cells_texts = [cell.text for cell in cells]
                formatted_cells = " | ".join(cells_texts)
                print(f"Following row is skipped: {formatted_cells}")
                continue

            entry = {
                "schedule title": self.schedule_title,
                "course number": course_id,
                "title": title,
                "units": units,
                "section": cells[3].text,
                "day": cells[4].text,
                "begin": cells[5].text,
                "end": cells[6].text,
                "room": cells[7].text,
                "location": cells[8].text,
                "instructor": cells[9].text
            }

            kv_texts = []
            for key, value in entry.items():
                kv_text = f"{key}: {value}"
                kv_texts.append(kv_text)
            entry_text = " | ".join(kv_texts)
            self._write_doc(entry_text)

        self._save_file()


if __name__ == "__main__":
    parser = ScheduleParser("https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_fall.htm",
                            "Fall 2024 Schedule")
    parser.parse()
