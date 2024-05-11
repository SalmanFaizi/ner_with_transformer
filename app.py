from flair.data import Sentence
from flair.models import SequenceTagger
import streamlit as st

# load tagger
tagger = SequenceTagger.load("flair/ner-english-large")

# make example sentence
text=st.text_area("Enter the text to detect it's named entities")
sentence = Sentence(text)

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print('The following NER tags are found:')
# iterate over entities and printx
for entity in sentence.get_spans('ner'):
    print(entity)


