# square_checkout.py

import os
from square.client import Client
from dotenv import load_dotenv
import uuid

load_dotenv()

# Initialize Square client
square_client = Client(
    access_token=os.getenv("SQUARE_ACCESS_TOKEN"),
    environment="production"  # or "sandbox"
)

location_id = os.getenv("SQUARE_LOCATION_ID")

def create_square_checkout(order_items):
    """Create a Square checkout link based on the order_items list."""

    # Prepare line items for the order
    line_items = []
    for item in order_items:
        line_items.append({
            "name": item,
            "quantity": "1",
            "base_price_money": {
                "amount": 1000,  # Set a placeholder price like $10.00 → 1000 cents
                "currency": "CAD"
            }
        })

    body = {
        "idempotency_key": str(uuid.uuid4()),
        "order": {
            "location_id": location_id,
            "line_items": line_items
        },
        "ask_for_shipping_address": False,
        "redirect_url": "https://pbx1-chatbot.onrender.com/thanks"
    }

    checkout_api = square_client.checkout
    response = checkout_api.create_checkout(location_id=location_id, body=body)

    if response.is_success():
        return response.body["checkout"]["checkout_page_url"]
    else:
        raise Exception(response.errors)
