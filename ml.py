import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("Apple buys U.K. startup for $1 billion.")
type(doc)

"""
For each of the entities in doc.ents, print the text and label of the corresponding entity
"""
for ent in doc.ents:
    print(ent.text, ent.label_)
