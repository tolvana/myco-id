import sys
import requests
from bs4 import BeautifulSoup

def get_mycological_characteristics(species):
    # Replace spaces with underscores for the Wikipedia URL
    species = species.replace(' ', '_')

    url = f'https://en.wikipedia.org/wiki/{species}'
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find the mycological characteristics infobox
        infoboxes = soup.find_all('table', class_='infobox')[1:]

        for infobox in infoboxes:

            if infobox is not None:
                characteristics = []
                # All 'tr' elements where data is stored in the infobox
                data_rows = infobox.find_all('tr')
                for row in data_rows:
                    # Finding label and data cells
                    label_cell = row.find('th', class_='infobox-label')
                    data_cell = row.find('td', class_='infobox-data')
                    if label_cell and data_cell:
                        # Extracting text and cleaning it
                        label = label_cell.get_text().strip()
                        data = data_cell.get_text().strip()
                        characteristics.append(data)
                if characteristics:
                    return characteristics
            else:
                print('Mycological characteristics infobox not found.')
    else:
        print('Failed to retrieve page:', url)

# Example usage:
species = sys.argv[1]
characteristics = get_mycological_characteristics(species)
if characteristics:
    for k in characteristics:
        print(k)
