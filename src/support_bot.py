"""
Support bot for the Ecommerce AI Communication Framework.

Generates a helpful solution/response for a customer query based on the issue
category and detected intent, and pairs it with the satisfaction prediction
so agents can see the likely customer sentiment.

This is a rule/keyword based assistant (no external LLM dependency) that maps
common e-commerce issues to actionable resolutions.
"""
import re

# Supported issue categories (aligned with the dataset categories).
SUPPORT_CATEGORIES = [
    "Order Related",
    "Returns",
    "Refund Related",
    "Cancellation",
    "Product Queries",
    "Payments related",
    "Feedback",
    "Shopzilla Related",
]

# Canned, actionable solutions per category.
CATEGORY_SOLUTIONS = {
    "Order Related": [
        "I can help with your order. You can track it in real time under **My Orders > Track**.",
        "If your order is delayed, most deliveries arrive within 3-5 business days. Delays during peak periods are common.",
        "If the tracking has not updated in 48 hours, I can raise a delivery investigation for you.",
    ],
    "Returns": [
        "You can start a return from **My Orders > Return Item** within the return window (usually 7-10 days of delivery).",
        "Please keep the original packaging and tags. A prepaid return label will be emailed to you.",
        "Once we receive the item, the return is processed within 2-3 business days.",
    ],
    "Refund Related": [
        "Refunds are issued to your original payment method after the returned item is received and inspected.",
        "Bank refunds typically take 5-7 business days; wallet/store-credit refunds are usually instant.",
        "If it has been more than 7 business days, I can escalate your refund for a priority review.",
    ],
    "Cancellation": [
        "You can cancel an order under **My Orders > Cancel** as long as it has not been shipped.",
        "If the order is already shipped, please refuse delivery or use the return flow once it arrives.",
        "Any amount paid is refunded automatically after a successful cancellation.",
    ],
    "Product Queries": [
        "You can find detailed specifications, size guides, and reviews on the product page.",
        "If a product is out of stock, use **Notify Me** to get an alert when it is restocked.",
        "For compatibility or usage questions, share the product name and I can pull the details.",
    ],
    "Payments related": [
        "We support cards, UPI, net banking, and wallets. If a payment failed, any deducted amount is auto-reversed within 3-5 business days.",
        "If you were charged but the order was not confirmed, please share the transaction ID so I can verify it.",
        "For EMI or coupon issues, make sure the offer conditions (minimum value, eligible cards) are met.",
    ],
    "Feedback": [
        "Thank you for sharing your feedback. I have recorded it for the relevant team.",
        "Your input helps us improve the shopping experience.",
        "If this feedback is about a specific order or agent, let me know and I can attach it to that case.",
    ],
    "Shopzilla Related": [
        "For account, membership, or platform questions, you can manage settings under **My Account**.",
        "Membership benefits like free delivery and early access apply automatically at checkout.",
        "If you are facing a login or app issue, clearing the cache or reinstalling usually resolves it.",
    ],
}

# Intent keywords that refine the response and set urgency.
INTENT_RULES = [
    (["not received", "never arrived", "missing", "lost", "where is"],
     "It sounds like a delivery issue. I will prioritise a delivery investigation on your order.", "high"),
    (["damaged", "broken", "defective", "not working", "faulty"],
     "I'm sorry the item arrived damaged. You are eligible for a free replacement or full refund.", "high"),
    (["refund not", "no refund", "refund pending", "still waiting", "not processed"],
     "I understand the refund is delayed. I am escalating this for a priority review.", "high"),
    (["wrong item", "different item", "incorrect"],
     "You received the wrong item. We will arrange a pickup and send the correct product at no cost.", "high"),
    (["cancel", "cancellation"],
     "I can help you cancel. If it has not shipped, cancellation is instant with an automatic refund.", "medium"),
    (["how", "when", "where", "can i", "what"],
     "Here is the information you need:", "low"),
    (["angry", "frustrated", "worst", "terrible", "disappointed", "unacceptable"],
     "I'm truly sorry for the frustration this caused. Let me make this right for you.", "high"),
]

CLOSINGS = {
    "high": "This has been flagged as urgent and a specialist will follow up shortly. Is there anything else I can help with?",
    "medium": "Let me know if you'd like me to proceed with this. Anything else I can help with?",
    "low": "I hope that helps! Is there anything else you'd like to know?",
}


def _detect_intent(query):
    """Return (intent_message, urgency) based on keywords in the query."""
    text = (query or "").lower()
    for keywords, message, urgency in INTENT_RULES:
        if any(k in text for k in keywords):
            return message, urgency
    return None, "low"


def generate_response(query, category):
    """
    Build a support response for a customer query.

    Returns a dict:
        {
          "greeting": str,
          "intent_message": str | None,
          "solution_steps": [str, ...],
          "closing": str,
          "urgency": "low"|"medium"|"high",
          "reply": str  # full assembled text
        }
    """
    category = category if category in CATEGORY_SOLUTIONS else "Order Related"
    intent_message, urgency = _detect_intent(query)
    solution_steps = CATEGORY_SOLUTIONS[category]
    closing = CLOSINGS[urgency]

    greeting = "Hi! Thanks for reaching out. "
    if not (query and query.strip()):
        greeting += "Please describe your issue and I'll do my best to help."
        return {
            "greeting": greeting,
            "intent_message": None,
            "solution_steps": [],
            "closing": "",
            "urgency": "low",
            "reply": greeting,
        }

    greeting += f"I see this is about **{category}**."

    # Assemble full reply text
    parts = [greeting]
    if intent_message:
        parts.append(intent_message)
    parts.append("Here's what you can do:")
    for i, step in enumerate(solution_steps, 1):
        parts.append(f"{i}. {step}")
    parts.append(closing)
    reply = "\n".join(parts)

    return {
        "greeting": greeting,
        "intent_message": intent_message,
        "solution_steps": solution_steps,
        "closing": closing,
        "urgency": urgency,
        "reply": reply,
    }
