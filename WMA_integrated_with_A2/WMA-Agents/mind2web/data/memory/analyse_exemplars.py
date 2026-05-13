import json

with open("exemplars.json", "r") as f:
    data = json.load(f)

print("Total top-level entries:", len(data))

# If nested, inspect first element
print("\nFirst element preview:\n")
import pprint
pprint.pprint(data[0])
