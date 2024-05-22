import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

base_url = "http://xangarsinfra.com"
start_url = base_url
data = []

def scrape_page(url):
    print(f"Scraping {url}")
    try:
        page = requests.get(url)
        if page.status_code == 404:
            return None
        soup = BeautifulSoup(page.text, "html.parser")
        return soup
    except Exception as e:
        print(f"Failed to retrieve {url}: {e}")
        return None

def parse_page(soup):
    page_data = {}
    # Extract data based on the actual structure of xangarsinfra.com
    # Example placeholders - replace with actual selectors
    page_data['title'] = soup.find('title').text if soup.find('title') else 'No Title'
    page_data['content'] = soup.find('div', class_='content').text.strip() if soup.find('div', class_='content') else 'No Content'
    # Add more fields as needed based on website structure

    data.append(page_data)

def scrape_website(start_url):
    to_visit = [start_url]
    visited = set()

    while to_visit:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        soup = scrape_page(current_url)
        if soup:
            parse_page(soup)
            visited.add(current_url)
            
            # Find and add new links to visit
            for link in soup.find_all('a', href=True):
                full_url = link['href']
                if not full_url.startswith('http'):
                    full_url = base_url + link['href']
                if full_url not in visited and full_url not in to_visit:
                    to_visit.append(full_url)

        # Be polite with a delay
        time.sleep(1)

# Start scraping from the homepage
scrape_website(start_url)

# Save the data to a file
df = pd.DataFrame(data)
df.to_excel("xangars_data.xlsx")
df.to_csv("xangars_data.csv")

print("Scraping completed and data saved.")