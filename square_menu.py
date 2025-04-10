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
    params = {"types": "ITEM,MODIFIER_LIST"}  # Add IMAGE if needed
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


if __name__ == "__main__":
    items, modifiers = get_catalog_items()
    print("Sample items:")
    for name, data in list(items.items())[:5]:
        print(f"- {name.title()} → ID: {data['id']}")
    print("\nSample modifiers:")
    for name, data in list(modifiers.items())[:3]:
        print(f"- {name.title()} → ID: {data['id']}")
