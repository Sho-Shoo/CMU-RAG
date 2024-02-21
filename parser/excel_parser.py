import pandas as pd

def extract_academic_cal(df):
    start_date_col = df.columns[0]
    end_date_col = df.columns[2]
    event_col = df.columns[-1]

    text_data = []

    for _, row in df.iterrows():
        if pd.isnull(row[start_date_col]):
            continue
        try:      
            start_date = pd.to_datetime(row[start_date_col])
            
            if pd.notnull(row[end_date_col]): # If the event spans multiple days
                text = []
                end_date = pd.to_datetime(row[end_date_col])
                num_days = (end_date - start_date).days + 1

                for day in range(num_days):
                    date = start_date + pd.DateOffset(days=day)
                    text.append(f"{date.strftime('%Y-%m-%d')} ")
                
                text.append(": ")
                text.append(row[event_col])        
                text_data.append("".join(text) + "<sep>\n") 

            else:
                text_data.append(f"{start_date.strftime('%Y-%m-%d')}: {row[event_col]}<sep>\n")

        except ValueError:
            continue
        
    return "".join(text_data)
    

def save_file(text, file_name):
    save_path = "knowledge_source/" + file_name.replace("/", "|")
    with open(save_path, 'w') as file:
        file.write(text)
        print(f"Text has been successfully saved to {save_path}")


if __name__ == "__main__":
    df_2324 = pd.read_excel("raw_data/raw_excel/2324-academic-calendar-list-view.xlsx")
    df_2425 = pd.read_excel("raw_data/raw_excel/2425-academic-calendar-list-view.xlsx")

    text_2324 = extract_academic_cal(df_2324.iloc[3:])
    text_2425 = extract_academic_cal(df_2425.iloc[3:])

    save_file(text_2324, "https://www.cmu.edu/hub/calendar/docs/2324-academic-calendar-list-view.txt")
    save_file(text_2425, "https://www.cmu.edu/hub/calendar/docs/2425-academic-calendar-list-view.txt")