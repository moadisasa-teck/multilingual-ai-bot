import re
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

queries = [
    "How long does it take to get a passport",
    "ፓስፖርት ለማውጣት ምን ሰነዶች ያስፈልጋሉ?",
    "Paaspoortii baasuuf maal barbaachisa"
]

for q in queries:
    print(f"Testing: {q}")
    # Regex check
    if re.search(r'[\u1200-\u137F]', q):
        print(" -> Regex: Amharic (Ethiopic script detected)")
        continue

    try:
        lang = detect(q)
        print(f" -> Langdetect: {lang}")
    except Exception as e:
        print(f" -> Error: {e}")
