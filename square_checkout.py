# square_checkout.py
import os
import uuid
from square.client import Client

client = Client(
    access_token=os.getenv("SQUARE_ACCESS_TOKEN"),
    environment="sandbox"  # Change to "production" when ready
)

def create_checkout(order_items, customer_email):
    body = {
        "idempotency_key": str(uuid.uuid4()),
        "order": {
            "location_id": os.getenv("SQUARE_LOCATION_ID"),
            "line_items": [
                {
                    "name": item["name"],
                    "quantity": str(item["quantity"]),
                    "base_price_money": {
                        "amount": int(float(item["price"]) * 100),
                        "currency": "CAD"
                    }
                }
                for item in order_items
            ],
        },
        "ask_for_shipping_address": True,
        "redirect_url": "https://yourdomain.com/thank-you"  # Replace this with your actual domain
    }

    response = client.checkout.create_checkout(
        location_id=os.getenv("SQUARE_LOCATION_ID"),
        body=body
    )

    return response.body["checkout"]["checkout_page_url"]
