import json
import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


st.markdown("# 📃 *tc")
st.markdown("### a tool that can read Terms & Conditions agreements and flag anything that is non-standard.")
tc = st.text_input("Enter the Terms & Conditions document")

if st.button("Analyze"):
    if tc=='':
        st.error("Please enter input T&C!")
    else:
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        pickle_in = open("embeddings.pkl","rb")
        stored_data=pickle.load(pickle_in)
        stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']

        obj = {}
        sentences = tc.split(". ")
        for i in sentences:
            if len(i) < 5:
                sentences.remove(i)
        embeddings = model.encode(sentences)
        for i in range(0, len(sentences)):
            arr = cosine_similarity([embeddings[i]], stored_embeddings[0:])
            if(np.max(arr) < 0.85):
                obj[sentences[i]] = np.max(arr)
                json_string = json.dumps(str(obj))
            tc="<style>.flag{color: red;}</style>"+tc
        for key in obj:
            substr=key
            beg="<div class='flag'><b>"
            end="</b></div>"
            idx = tc.index(substr)
            temp = idx+len(substr)
            tc=tc[:idx] + beg + substr + end + tc[temp:]
        st.success("Found a few non-standard clauses. They're flagged below.")
        st.markdown(tc, unsafe_allow_html=True)


