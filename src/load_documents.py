from pathlib import Path
from pypdf import PdfReader

data_folder = Path("data")

for pdf_path in data_folder.glob("*.pdf"):
    print(f"\nLecture de : {pdf_path.name}")

    reader = PdfReader(pdf_path)

    print(f"Nombre de pages : {len(reader.pages)}")

    first_page_text = reader.pages[0].extract_text()

    print("\nDébut du texte :")
    print(first_page_text[:500])

    from pathlib import Path
from pypdf import PdfReader

data_folder = Path("data")

documents = []

for pdf_path in data_folder.glob("*.pdf"):
    print(f"Lecture de : {pdf_path.name}")

    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            documents.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": text
            })

print("\nExtraction terminée !")
print("Nombre total de pages extraites :", len(documents))

print("\nExemple :")
print("Source :", documents[0]["source"])
print("Page :", documents[0]["page"])
print("Texte :", documents[0]["text"][:500])