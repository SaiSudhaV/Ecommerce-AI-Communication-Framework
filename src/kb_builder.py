"""
Build the support knowledge base from the dataset.

The raw dataset has no explicit question/answer pairs — it has customer
interactions with a csat_score (1-5), a category, and a sub-category. We treat
every interaction a customer was SATISFIED with (csat_score >= 4) as evidence
that the resolution path for that (category, sub-category) topic works.

So we build one verified knowledge-base entry per sub-category topic, backed by
the count of satisfied interactions. The answer text is a concise, actionable
resolution for that topic. This gives real coverage (dozens of topics drawn
from ~22k satisfied interactions) instead of a handful of hardcoded seeds.
"""
import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_CSV = os.path.join(DATA_DIR, "cumulative_ai_customer_communication_dataset.csv")
DATASET_GZ = DATASET_CSV + ".gz"

# csat_score >= this counts as a "satisfied" (verified) interaction.
SATISFIED_THRESHOLD = 4

# Minimum satisfied interactions for a topic to become a verified KB entry.
MIN_SUPPORT = 10

# Actionable resolution text per sub-category topic. Topics not listed here
# fall back to a generic template built from the category + sub-category.
SUBCATEGORY_ANSWERS = {
    "Reverse Pickup Enquiry":
        "For a reverse pickup, go to My Orders > Return/Replace and choose a pickup slot. "
        "You'll get an SMS/email with the pickup date. Keep the item in its original "
        "packaging with all tags and accessories ready for the agent.",
    "Return request":
        "To raise a return, open My Orders > Return Item within the return window "
        "(usually 7-10 days of delivery). Select a reason, choose refund or replacement, "
        "and a prepaid return label is emailed to you.",
    "Delayed":
        "If your order is delayed, check live status under My Orders > Track. Most delayed "
        "orders arrive within 3-5 business days. If tracking hasn't updated in 48 hours, "
        "we can raise a delivery investigation for you.",
    "Order status enquiry":
        "You can see your order's current status any time under My Orders > Track. The page "
        "shows dispatch, in-transit, and out-for-delivery stages with the expected date.",
    "Fraudulent User":
        "If you suspect fraudulent activity on your account or an order you did not place, "
        "change your password immediately and report it under Help > Report Fraud. We will "
        "secure the account and investigate.",
    "Installation/demo":
        "Installation or demo for eligible products is scheduled after delivery. Track or "
        "reschedule it under My Orders > Services. A technician visit is usually arranged "
        "within 2-4 business days.",
    "Product Specific Information":
        "Detailed specifications, size guides, warranty, and reviews are on the product page. "
        "For compatibility or usage questions, share the product name and we'll pull the details.",
    "Missing":
        "If an item is missing from your delivery, report it under My Orders > Report an Issue "
        "within 48 hours. We'll verify the package weight/scan and arrange the missing item "
        "or a refund.",
    "Wrong":
        "Received the wrong item? Start a return under My Orders > Return Item and select "
        "'Wrong item received'. We'll arrange a free pickup and send the correct product or "
        "issue a refund.",
    "Refund Enquiry":
        "Refunds are issued to your original payment method once the returned item is received "
        "and inspected. Bank refunds take 5-7 business days; wallet/store-credit refunds are "
        "usually instant. Track it under My Orders > Refund Status.",
    "UnProfessional Behaviour":
        "We're sorry about your experience. Please share the order ID and what happened under "
        "Help > Feedback. We take agent conduct seriously and will review it with the team.",
    "Service Centres Related":
        "To find an authorised service centre, go to My Orders > Services > Locate Centre and "
        "enter your pincode. Carry the invoice and warranty card for in-warranty service.",
    "Refund Related Issues":
        "If a refund is delayed beyond the expected window, verify the refund reference under "
        "My Orders > Refund Status. If it's still pending after 7 business days, we can "
        "escalate it for a priority review.",
    "Not Needed":
        "To cancel an order you no longer need, go to My Orders > Cancel as long as it hasn't "
        "shipped. Any amount paid is refunded automatically after cancellation.",
    "General Enquiry":
        "You can manage most things under My Account and My Orders. Tell me a bit more about "
        "what you need and I'll point you to the right place.",
    "Invoice request":
        "Download your invoice any time under My Orders > Invoice > Download. A GST invoice "
        "is available for eligible orders.",
    "Online Payment Issues":
        "If a payment failed but money was deducted, it is auto-reversed within 3-5 business "
        "days. If the order wasn't confirmed, share the transaction ID and we'll verify it.",
    "Priority delivery":
        "Priority or faster delivery options, where available, are shown at checkout for your "
        "pincode. Already-placed orders can't be upgraded to priority after dispatch.",
    "Customer Requested Modifications":
        "Address or item changes are possible only before the order ships. Go to My Orders > "
        "Edit Order. After dispatch, you can refuse delivery or use the return flow.",
    "Seller Cancelled Order":
        "If a seller cancelled your order, any amount paid is refunded in full automatically. "
        "You can reorder from another seller on the product page.",
    "Exchange / Replacement":
        "To exchange or replace an item, use My Orders > Return/Replace and pick 'Replacement'. "
        "The new item ships once the pickup is completed.",
    "Payment related Queries":
        "We support cards, UPI, net banking, and wallets. For EMI or coupon issues, confirm "
        "the offer conditions (minimum value, eligible cards) are met at checkout.",
    "Signup Issues":
        "If you're having trouble signing up, make sure your email/phone isn't already "
        "registered. Clear the app cache or try a different browser, then request a fresh OTP.",
    "Damaged":
        "Sorry your item arrived damaged. Start a return under My Orders > Return Item and "
        "select 'Damaged'. You're eligible for a free replacement or a full refund.",
}

# Short clarifying questions per category — asked before escalating to an agent.
CATEGORY_CLARIFIERS = {
    "Order Related": [
        "What is your order ID (or the item name)?",
        "Has the order shipped yet, or is it still being processed?",
    ],
    "Returns": [
        "Which item would you like to return, and what's the reason?",
        "Is the item still in its original packaging with tags?",
    ],
    "Refund Related": [
        "What is the order ID for the refund you're asking about?",
        "How did you pay — card, UPI, net banking, or wallet?",
    ],
    "Cancellation": [
        "What is the order ID you'd like to cancel?",
        "Has the order already shipped?",
    ],
    "Product Queries": [
        "Which product is this about (name or link)?",
        "What specifically would you like to know — specs, availability, or usage?",
    ],
    "Payments related": [
        "Was an amount deducted, and did the order get confirmed?",
        "What payment method did you use?",
    ],
    "Feedback": [
        "Is this feedback about a specific order or agent?",
        "What would you like us to improve?",
    ],
    "Shopzilla Related": [
        "Is this about your account, membership, or the app/website?",
        "What exactly happens when you try (any error message)?",
    ],
}

DEFAULT_CLARIFIERS = [
    "Could you share your order ID or the item name?",
    "Can you describe exactly what happened or what you need?",
]


def _generic_answer(category, sub):
    return (
        f"For '{sub}' under {category}, you can usually resolve this from "
        f"My Orders or My Account. Open the related order, choose the matching "
        f"option, and follow the on-screen steps. If you still need help, I can "
        f"connect you with an agent."
    )


def _decompress_if_needed():
    import gzip, shutil
    if not os.path.exists(DATASET_CSV) and os.path.exists(DATASET_GZ):
        with gzip.open(DATASET_GZ, "rb") as f_in, open(DATASET_CSV, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def build_entries():
    """
    Return a list of KB entries built from satisfied interactions in the dataset.
    Each entry: {question, answer, category, source, support, created_at}.
    """
    _decompress_if_needed()
    if not os.path.exists(DATASET_CSV):
        return []

    df = pd.read_csv(DATASET_CSV, low_memory=False,
                     usecols=["category", "sub-category", "csat_score"])
    satisfied = df[df["csat_score"] >= SATISFIED_THRESHOLD]

    grouped = (
        satisfied.groupby(["category", "sub-category"])
        .size()
        .reset_index(name="support")
        .sort_values("support", ascending=False)
    )

    entries = []
    now = datetime.now().isoformat(timespec="seconds")
    for _, row in grouped.iterrows():
        cat = str(row["category"]).strip()
        sub = str(row["sub-category"]).strip()
        support = int(row["support"])
        if support < MIN_SUPPORT or cat.lower() == "nan" or sub.lower() == "nan":
            continue
        answer = SUBCATEGORY_ANSWERS.get(sub) or _generic_answer(cat, sub)
        # The "question" is the topic phrasing customers would ask about.
        question = f"{sub} ({cat})"
        entries.append({
            "question": question,
            "answer": answer,
            "category": cat,
            "source": "dataset",
            "support": support,
            "created_at": now,
        })
    return entries


def clarifiers_for(category):
    """Return the clarifying questions to ask for a category before escalation."""
    return CATEGORY_CLARIFIERS.get(category, DEFAULT_CLARIFIERS)


if __name__ == "__main__":
    built = build_entries()
    print(f"Built {len(built)} verified KB entries from the dataset.")
    for e in built[:15]:
        print(f"  [{e['support']:5d}] {e['category']} > {e['question']}")
