# order.py

from typing import List, Dict

# Global order tracker
order = []

# Tax rates
GST = 0.05  # Goods and Services Tax (5%)
PST = 0.07  # Provincial Sales Tax (7%) for drinks/dips only

# Menu prices and categories for tax logic
menu_items = {
    "Caesar Salad (Regular)": {"price": 7.00, "type": "food"},
    "Caesar Salad (Large)": {"price": 10.00, "type": "food"},
    "Pop (355ml)": {"price": 2.00, "type": "beverage"},
    "Pop (591ml)": {"price": 2.80, "type": "beverage"},
    "Pop (2L)": {"price": 4.40, "type": "beverage"},
    "Rockstar Energy Drink": {"price": 4.00, "type": "beverage"},
    "Kraft Ranch Dip": {"price": 0.25, "type": "dip"},
    "Hellsmann Dip": {"price": 1.00, "type": "dip"},
}

def add_to_order(item: str) -> str:
    if item in menu_items:
        order.append(item)
        return f"✅ Added {item} to your order."
    else:
        return f"❌ Sorry, {item} is not on the menu."

def clear_order():
    order.clear()

def get_order() -> List[str]:
    return order

def generate_order_summary() -> str:
    if not order:
        return "🧾 Your order is currently empty."

    subtotal = 0
    gst_total = 0
    pst_total = 0
    lines = ["\n🧾 Your Order Summary:\n"]

    for item in order:
        details = menu_items.get(item, {})
        price = details.get("price", 0)
        category = details.get("type", "food")
        subtotal += price
        gst = price * GST
        pst = price * PST if category in ("beverage", "dip") else 0
        gst_total += gst
        pst_total += pst
        lines.append(f"- {item:<40} ${price:.2f}")

    total = round(subtotal + gst_total + pst_total, 2)

    lines.append(f"\nSubtotal:{'':<34} ${subtotal:.2f}")
    lines.append(f"GST (5%):{'':<34} ${gst_total:.2f}")
    lines.append(f"PST (7% on drinks/dips):{'':<19} ${pst_total:.2f}")
    lines.append(f"\n🎯 Total:{'':<34} ${total:.2f}\n")

    return "\n".join(lines)