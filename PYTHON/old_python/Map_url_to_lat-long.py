import requests
import re
import csv
marion_urls = [
    "https://maps.app.goo.gl/jBA2S8HKicUxmtpS7",
    "https://maps.app.goo.gl/6E3UbPe8LejeHdDcA",
    "https://maps.app.goo.gl/zsxo5gKGffUH3MYo8",
    "https://maps.app.goo.gl/AXnckMfs3LQexa7t6",
    "https://maps.app.goo.gl/oJck82udbFo1ahwb8",
    "https://maps.app.goo.gl/3awnE5D8wVBYvDvc8",
    "https://maps.app.goo.gl/HnQVdxu8PrRAPtEb7",
    "https://maps.app.goo.gl/koERijcyCy2ZXsJT8",
    "https://maps.app.goo.gl/DCLXNEaA7GsZCstdA",
    "https://maps.app.goo.gl/7XmWysmeMWQpATrF6",
    "https://maps.app.goo.gl/ABEyfwFk2tjSkqKp6",
    "https://maps.app.goo.gl/owNyF51HbMn4kHHH9",
    "https://maps.app.goo.gl/LZpzmYz8g8U2Mm9X6",
    "https://maps.app.goo.gl/UwDPFC9UcXbB6r8QA",
    "https://maps.app.goo.gl/i1HgybFZzhewZUPB9"
]

urls = [
    "https://maps.app.goo.gl/KBezRHHb77Kw8wEg9",
    "https://maps.app.goo.gl/QF5suMW4ekpVQFQFA",
    "https://maps.app.goo.gl/ngft7ZrHbuE4sWGA6",
    "https://maps.app.goo.gl/HH8NSdrDx6KFkkPS8",
    "https://maps.app.goo.gl/ErCHJ6zfgMxpL55E8",
    "https://maps.app.goo.gl/bH8hpcezW8dPfxzB6"
]




# Regex patterns for extracting lat/lng
pattern_at     = re.compile(r'@(-?\d+\.\d+),(-?\d+\.\d+)')
pattern_query  = re.compile(r'[?&]query=(-?\d+\.\d+),(-?\d+\.\d+)')
pattern_center = re.compile(r'[?&](?:ll|center)=(-?\d+\.\d+),(-?\d+\.\d+)')
pattern_search = re.compile(r'/search/(-?\d+\.\d+),\+?(-?\d+\.\d+)')

def get_lat_lng(share_url):
    """
    Follows redirects of the short URL and extracts lat/lng
    from the final Google Maps URL.
    Returns (lat, lng) or (None, None) on failure.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(share_url, headers=headers, allow_redirects=True, timeout=10)
        final_url = resp.url
    except Exception:
        return None, None

    for pattern in (pattern_at, pattern_query, pattern_center, pattern_search):
        m = pattern.search(final_url)
        if m:
            return float(m.group(1)), float(m.group(2))

    return None, None

if __name__ == "__main__":
    # Prepare CSV output
    with open("coordinates.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["url", "latitude", "longitude"])

        for url in urls:
            lat, lng = get_lat_lng(url)
            writer.writerow([url, lat if lat is not None else "", lng if lng is not None else ""])

    print("Exported coordinates to coordinates.csv")

