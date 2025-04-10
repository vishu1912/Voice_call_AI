import os
from square.client import Client
from dotenv import load_dotenv
import uuid

load_dotenv()

# Initialize Square client
square_client = Client(
    access_token=os.getenv("SQUARE_ACCESS_TOKEN"),
    environment="production"
)

location_id = os.getenv("SQUARE_LOCATION_ID")

# Your Square menu should be structured like this:
# {
#     "Garlic Toast": {"variation_id": "ABC123", "price": 699},
#     "Pizza": {"variation_id": "XYZ789", "price": 1599}
# }

def create_square_checkout(order_items: list[str], square_menu: dict) -> str:
    """Create a Square Checkout URL based on actual catalog data."""

    line_items = []

    for item_name in order_items:
        item_data = square_menu.get(item_name)
        if item_data:
            line_items.append({
                "catalog_object_id": item_data["variation_id"],
                "quantity": "1"
            })
        else:
            raise Exception(f"Menu item not found in catalog: {item_name}")

    if not line_items:
        raise Exception("No valid items in the order.")

    body = {
        "idempotency_key": str(uuid.uuid4()),
        "order": {
            "location_id": location_id,
            "line_items": line_items
        },
        "redirect_url": "https://pbx1-chatbot.onrender.com/thanks"
    }

    response = square_client.checkout.create_checkout(location_id=location_id, body=body)

    if response.is_success():
        return response.body["checkout"]["checkout_page_url"]
    else:
        raise Exception(response.errors)


# ✅ Add this function for delivery validation
def is_address_deliverable(address: str) -> bool:
    """Fake delivery zone validator using a basic postal code match (you can customize this)."""
    # Add your deliverable postal codes here (get from Square if needed)
    valid_postal_prefixes = ["V2S", "V2T", "V3G", "V4X"]

    for prefix in valid_postal_prefixes:
        if prefix.lower() in address.lower().replace(" ", ""):
            return True
    return False
