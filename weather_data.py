import os
import zipfile
import requests
from io import BytesIO
from bs4 import BeautifulSoup
import csv
import pandas as pd

years = ["2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016", "2015", "2014", "2013"]
base_url = "https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/dobowe/synop/"
output_file = "weather_data_full.csv"
summary_file = "weather_data_selected.csv"

columns = [
    "Station Code", "Station Name", "Year", "Month", "Day",
    "Average Daily Total Cloud Cover [oktas]", "NOS Measurement Status",
    "Average Daily Wind Speed [m/s]", "FWS Measurement Status",
    "Average Daily Temperature [°C]", "TEMP Measurement Status",
    "Average Daily Water Vapor Pressure [hPa]", "CPW Measurement Status",
    "Average Daily Relative Humidity [%]", "WLGS Measurement Status",
    "Average Daily Station-Level Pressure [hPa]", "PPPS Measurement Status",
    "Average Daily Sea-Level Pressure [hPa]", "PPPM Measurement Status",
    "Daily Precipitation Sum [mm]", "WODZ Measurement Status",
    "Night Precipitation Sum [mm]", "WONO Measurement Status"
]

summary_columns = [
    "Date",
    "Average Daily Total Cloud Cover [oktas]",
    "Average Daily Wind Speed [m/s]",
    "Average Daily Temperature [°C]",
    "Average Daily Relative Humidity [%]",
    "Average Daily Station-Level Pressure [hPa]",
    "Daily Precipitation Sum [mm]",
    "Night Precipitation Sum [mm]"
]

def download_and_extract_zip(url, year):
    try:
        response = requests.get(url)
        response.raise_for_status()

        if 'application/zip' in response.headers.get('Content-Type', ''):
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                target_filename = f"s_d_t_155_{year}.csv"

                if target_filename in zf.namelist():
                    zf.extract(target_filename, "temp")
                    print(f"Rozpakowano plik: {target_filename}")
                    return os.path.join("temp", target_filename)
                else:
                    print(f"Brak pliku {target_filename} w ZIP {url}")
        else:
            print(f"Plik na URL {url} nie jest ZIP-em.")
    except Exception as e:
        print(f"Błąd pobierania: {e}")

    return None

def get_files_from_folder(url, year):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        zip_filename = f"{year}_155_s.zip"
        files = [link['href'] for link in soup.find_all('a') if zip_filename in link.get('href', '')]
        return files
    except Exception as e:
        print(f"Błąd dostępu do folderu {url}: {e}")
        return []

def process_and_append_csv(file_path):
    try:
        with open(file_path, 'r', encoding='latin1') as file, \
             open(output_file, 'a', newline='', encoding='utf-8') as csvfile, \
             open(summary_file, 'a', newline='', encoding='utf-8') as summary_csvfile:

            reader = csv.reader(file)
            writer = csv.writer(csvfile)
            summary_writer = csv.writer(summary_csvfile)

            for row in reader:

                writer.writerow(row)

                date = f"{row[2]}-{row[3].zfill(2)}-{row[4].zfill(2)}"

                summary_row = [
                    date,
                    row[5],     #Średnie dobowe zachmurzenie ogólne [oktanty]
                    row[7],     #Średnia dobowa prędkość wiatru [m/s]
                    row[9],     #Średnia dobowa temperatura [°C]
                    row[13],    #Średnia dobowa wilgotność względna [%]
                    row[15],    #Średnie dobowe ciśnienie na poziomie stacji [hPa]
                    row[19],    #Suma opadu dzień [mm]",
                    row[21]     #Suma opadu noc [mm]",
                ]

                summary_writer.writerow(summary_row)

        print(f"Dodano dane z {file_path} do {output_file} i {summary_file}")
    except Exception as e:
        print(f"Błąd przetwarzania pliku {file_path}: {e}")

def sort_and_save_data():
    try:
        full_data = pd.read_csv(output_file)

        if full_data.isnull().any().any():
            print("Dane zawierają brakujące wartości (NaN).")
            full_data = full_data.fillna(0)

        full_data['Date'] = pd.to_datetime(full_data['Year'].astype(str) + '-' + full_data['Month'].astype(str).str.zfill(2) + '-' + full_data['Day'].astype(str).str.zfill(2))
        full_data = full_data.sort_values(by='Date')
        full_data.to_csv(output_file, index=False, encoding='utf-8')

        summary_data = pd.read_csv(summary_file)
        summary_data['Date'] = pd.to_datetime(summary_data['Date'])
        summary_data = summary_data.sort_values(by='Date')
        summary_data.to_csv(summary_file, index=False, encoding='utf-8')

        print(f"Dane zostały posortowane i zapisane do: {output_file} oraz {summary_file}")
    except Exception as e:
        print(f"Błąd przy sortowaniu i zapisywaniu danych: {e}")

def main():

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)

    with open(summary_file, 'w', newline='', encoding='utf-8') as summary_csvfile:
        summary_writer = csv.writer(summary_csvfile)
        summary_writer.writerow(summary_columns)

    for year in years:
        folder_url = f"{base_url}{year}/"
        print(f"Sprawdzam folder: {folder_url}")

        files = get_files_from_folder(folder_url, year)
        if files:
            for filename in files:
                zip_url = f"{folder_url}{filename}"
                extracted_csv = download_and_extract_zip(zip_url, year)

                if extracted_csv:
                    process_and_append_csv(extracted_csv)
        else:
            print(f"Brak plików w folderze {folder_url}")

    sort_and_save_data()


    print(f"Wszystkie dane zapisano do: {output_file} i {summary_file}")

if __name__ == "__main__":
    main()
