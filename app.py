import os
import streamlit as st
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

# Set Hugging Face API token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# Backend functions
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt,
                     llm=HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1",
                                        model_kwargs={"temperature": 0,
                                                      "max_length": 64}))

history = {'question': [], 'answers': []}


def LLm(question):
    answer = llm_chain.run(question)
    history['question'].append(question)
    history['answers'].append(answer)
    return history


# Streamlit app
st.title("Stonks: Your Paisaa Vasool Pal!!!")

# User input
user_input = st.text_input("What is your question?:", "")

# Process input and get response
if st.button("Send"):
    if user_input:
        history = LLm(user_input)
        st.write(f"You: {user_input}")
        st.write(f"Bot: {history['answers'][-1]}")

