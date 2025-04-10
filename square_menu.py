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
    params = {"types": "ITEM,MODIFIER_LIST"}
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()

def get_catalog_items():
    catalog = fetch_square_catalog()
    items = {}
    modifiers = {}

    for obj in catalog.get("objects", []):
        if obj["type"] == "ITEM":
            name = obj["item_data"]["name"].lower()
            items[name] = {
                "id": obj["id"],
                "variations": obj["item_data"].get("variations", [])
            }
        elif obj["type"] == "MODIFIER_LIST":
            name = obj["modifier_list_data"]["name"].lower()
            modifiers[name] = {
                "id": obj["id"],
                "modifiers": obj["modifier_list_data"].get("modifiers", [])
            }

    return items, modifiers

def get_square_menu_items():
    items, _ = get_catalog_items()
    simplified_menu = {}

    for name, data in items.items():
        variations = data.get("variations", [])
        prices = []
        for v in variations:
            var_data = v.get("item_variation_data", {})
            price_money = var_data.get("price_money", {})
            price = price_money.get("amount", 0) / 100
            currency = price_money.get("currency", "CAD")
            prices.append(f"{price:.2f} {currency}")
        simplified_menu[name.title()] = prices

    return simplified_menu

if __name__ == "__main__":
    menu = get_square_menu_items()
    print("Sample Menu Items:")
    for item, prices in list(menu.items())[:5]:
        print(f"- {item}: {', '.join(prices)}")
