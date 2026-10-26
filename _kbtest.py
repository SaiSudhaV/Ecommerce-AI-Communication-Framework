from src import knowledge_base as kb

tests = [
    ("reverse pickup enquiry", "Returns"),
    ("return request", "Returns"),
    ("delayed", "Order Related"),
    ("refund enquiry", "Refund Related"),
    ("I want to return my item", "Returns"),
    ("my refund is not processed", "Refund Related"),
]
for q, c in tests:
    r = kb.lookup(q, category=c)
    print(f"{q!r:34} [{c:16}] -> matched={r['matched']} score={r['score']}")
