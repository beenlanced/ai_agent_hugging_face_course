# Insert data from metadata.jsonl into supabase 
# Follow guidance from: 
# - https://python.langchain.com/docs/integrations/vectorstores/supabase/0

from collections import Counter, OrderedDict
import json
import os
import random

from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from supabase.client import Client, create_client


# Load the metadata.jsonl file
with open("metadata.jsonl", 'r') as jsonl_file:
    json_list = list(jsonl_file)

json_QA = []
for json_str in json_list:
    json_data = json.loads(json_str)
    json_QA.append(json_data)

# Check that I was able to load data and extract everything appropriately
# randomly select 1 samples
# random.seed(42)
random_samples = random.sample(json_QA, 1)
for sample in random_samples:
    print("=" * 50)
    print(f"Task ID: {sample['task_id']}")
    print(f"Question: {sample['Question']}")
    print(f"Level: {sample['Level']}")
    print(f"Final Answer: {sample['Final answer']}")
    print(f"Annotator Metadata: ")
    print(f"  ├── Steps: ")
    for step in sample['Annotator Metadata']['Steps'].split('\n'):
        print(f"  │      ├── {step}")
    print(f"  ├── Number of steps: {sample['Annotator Metadata']['Number of steps']}")
    print(f"  ├── How long did this take?: {sample['Annotator Metadata']['How long did this take?']}")
    print(f"  ├── Tools:")
    for tool in sample['Annotator Metadata']['Tools'].split('\n'):
        print(f"  │      ├── {tool}")
    print(f"  └── Number of tools: {sample['Annotator Metadata']['Number of tools']}")
print("=" * 50)


#####
# Writing out the prompt
#####
system_prompt = """
You are a helpful assistant tasked with answering questions using a set of tools.
If the tool is not available, you can try to find the information online. You can also use your own knowledge to answer the question. 
You need to provide a step-by-step explanation of how you arrived at the answer.
==========================
Here is a few examples showing you how to answer the question step by step.
"""
for i, samples in enumerate(random_samples):
    system_prompt += f"\nQuestion {i+1}: {samples['Question']}\nSteps:\n{samples['Annotator Metadata']['Steps']}\nTools:\n{samples['Annotator Metadata']['Tools']}\nFinal Answer: {samples['Final answer']}\n"
system_prompt += "\n==========================\n"
system_prompt += "Now, please answer the following question step by step.\n"

print("\n")
print(system_prompt)
print("\n")

#save the system_prompt to a file - commenting code as I have saved the system prompt
# with open('system_prompt.txt', 'w') as f:
#     f.write(system_prompt)

#####
# List of the tools used in all the samples
#   Helps me understand what tools I will need to make available to my agent
#####
tools = []
for sample in json_QA:
    for tool in sample['Annotator Metadata']['Tools'].split('\n'):
        tool = tool[2:].strip().lower()
        if tool.startswith("("):
            tool = tool[11:].strip()
        tools.append(tool)
tools_counter = OrderedDict(Counter(tools))
print("List of tools used in all samples:")
print("Total number of tools used:", len(tools_counter))
for tool, count in tools_counter.items():
    print(f"  ├── {tool}: {count}")


# Get API KEYS 
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

# Create supabase Client
supabase: Client = create_client(supabase_url, supabase_key)

# Set embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #  dim=768

#####
# Wrap the metadata.jsonl's questions and answers into a list of documents
#####

# docs = []
# for sample in json_QA:
#     content = f"Question : {sample['Question']}\n\nFinal answer : {sample['Final answer']}"
#     doc = {
#         "content" : content,
#         "metadata" : {
#             "source" : sample['task_id']
#         },
#         "embedding" : embeddings.embed_query(content),
#     }
#     docs.append(doc)

# Upload the documents to the supabase vector database - Commented out as I have previously done this step
# try:
#     response = (
#         supabase.table("documents")
#         .insert(docs)
#         .execute()
#     )
# except Exception as exception:
#     print("Error inserting data into Supabase:", exception)

#########
# Verfiy that I can extract data from supabase table: documents
########

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding= embeddings,
    table_name="documents",
    query_name="match_documents",
)
retriever = vector_store.as_retriever()

query = "On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article mentions a team that produced a paper about their observations, linked at the bottom of the article. Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?"
docs = retriever.invoke(query)
print("\n", docs[0])