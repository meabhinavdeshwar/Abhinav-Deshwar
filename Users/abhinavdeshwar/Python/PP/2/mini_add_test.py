"""Simple smoke test for add-row logic."""


from utils import parse_rows

rows = []
rows.append({"year": 2023, "month": 1, "demand": 10})
rows.append({"year": 2023, "month": 2, "demand": 12})

parsed = parse_rows(rows)
print(parsed)

