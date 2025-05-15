import requests
import csv
import os
from datetime import datetime

url = "https://www.szadolki.pl/stenchmap/points_get.php"

headers = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Mobile Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://www.szadolki.pl",
    "Referer": "https://www.szadolki.pl/mapa_smrodu.php"
}

input_file = "smell_data_full.csv"
output_file = "smell_data_selected.csv"

def get_last_modified_timestamp(file_path=input_file):
    if not os.path.exists(file_path):
        return None

    last_modified_timestamp = os.path.getmtime(file_path)
    last_modified_datetime = datetime.fromtimestamp(last_modified_timestamp)

    return last_modified_datetime.strftime("%Y-%m-%d %H:%M:%S")
def fetch_and_save_data():
    now = datetime.now()

    last_modified_timestamp = get_last_modified_timestamp()

    if last_modified_timestamp:
        start_date = last_modified_timestamp
    else:
        start_date = (now.replace(year=now.year - 20)).strftime("%Y-%m-%d %H:%M:%S")
    end_date = now.strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "from": start_date,
        "to": end_date
    }

    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Pobieram dane od {start_date} do {end_date}...")

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        try:
            json_data = response.json()

            if isinstance(json_data, list) and json_data:
                with open(input_file, "a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)

                    if os.stat(input_file).st_size == 0:
                        writer.writerow(json_data[0].keys())

                    for record in json_data:
                        writer.writerow(record.values())

                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Zapisano {len(json_data)} rekordów do smell_data_full.csv")
            else:
                print("Brak nowych danych lub nieoczekiwany format.")
        except requests.exceptions.JSONDecodeError:
            print(f"Błąd dekodowania JSON. Serwer zwrócił: {response.text}")
    else:
        print(f"Błąd w zapytaniu. Status: {response.status_code}")


fetch_and_save_data()

selected_columns = ["lat", "lng", "level", "observed"]

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", newline="", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=selected_columns)

    writer.writeheader()

    for row in reader:
        filtered_row = {col: row[col] for col in selected_columns}
        writer.writerow(filtered_row)

print(f"Zapisano wybrane kolumny do pliku: {output_file}")