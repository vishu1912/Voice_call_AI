# square_menu.py
import os
import requests

SQUARE_ACCESS_TOKEN = os.getenv("SQUARE_ACCESS_TOKEN")
SQUARE_LOCATION_ID = os.getenv("SQUARE_LOCATION_ID")

HEADERS = {
    "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

BASE_URL = "https://connect.squareup.com/v2"

def fetch_square_catalog():
    url = f"{BASE_URL}/catalog/list"
    params = {"types": "ITEM,MODIFIER_LIST,IMAGE"}
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()

def get_catalog_items():
    catalog = fetch_square_catalog()
    items = {}
    for obj in catalog.get("objects", []):
        if obj["type"] == "ITEM":
            name = obj["item_data"]["name"].lower()
            variation = obj["item_data"]["variations"][0]  # take first variation
            variation_id = variation["id"]
            price_data = variation["item_variation_data"].get("price_money", {})
            price = price_data.get("amount", 0)
            currency = price_data.get("currency", "CAD")
            items[name] = {
                "variation_id": variation_id,
                "price": price,
                "currency": currency
            }
    return items

def get_square_menu_items(full_data=False):
    items = get_catalog_items()
    if full_data:
        return items
    simplified = {}
    for name, data in items.items():
        simplified[name.title()] = [f"{data['price'] / 100:.2f} {data['currency']}"]
    return simplified

if __name__ == "__main__":
    sample = get_square_menu_items()
    print("Sample menu:")
    for k, v in list(sample.items())[:5]:
        print(f"- {k}: {v}")
