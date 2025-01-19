
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
import openai
from bertopic.representation import LangChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from bertopic.representation import OpenAI
import os
import dotenv


dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Load the dataset
dataset = pd.read_excel('data/contraints_topics.xlsx')
sentences = dataset['sentence'].tolist()

# Get embeddings from the pkl file
# You can extract the embeddings from the ClimateBERT-Promise_Constraint model
embeddings = pd.read_pickle('embeddings.pkl')


# LangChain representation model
chain = load_qa_chain(OpenAI(temperature=0, 
                             openai_api_key=OPENAI_API_KEY), 
                            chain_type="stuff")
prompt = 'You are given the following sentences: [DOCUMENTS] \nProvide only a label without any explanation. You can choose from the following labels: "Economic Restrictions", "Geopolitical Conditions", "Operational Obstacles", "Market Risks", "Human Capital", "Other Impacts"'
langchain = LangChain(chain, prompt=prompt)


# OpenAI representation model
client = openai.OpenAI(api_key=OPENAI_API_KEY)
prompt = 'You are given the following sentences: [DOCUMENTS] \nProvide only a label without any explanation. You can choose from the following labels: "Economic Restrictions", "Geopolitical Conditions", "Operational Obstacles", "Market Risks", "Human Capital", "Other Impacts"'
openai_model = OpenAI(client, model="gpt-3.5-turbo", chat=True, prompt=prompt)


representation_model = {"openai_representation": openai_model,
                        "langchain": langchain}

# Reduce dimensionality
umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

topic_model = BERTopic(calculate_probabilities=True, 
                       verbose=True,
                       umap_model=umap_model, 
                       min_topic_size=10, 
                       nr_topics="auto",
                       representation_model=representation_model)
# Fit the model on your documents with precomputed embeddings
topics, probs = topic_model.fit_transform(sentences, 
                                          embeddings
                                          )


# Get the distribution of topics
topic_information = topic_model.get_topic_info()


# Save to the csv file
topic_information.to_csv('topics_information.csv', index=False)


