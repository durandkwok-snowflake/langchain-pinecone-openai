# Langchain-Pinecone-Openai

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/langchain-retrieval-agent.ipynb) [![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/pinecone-io/examples/blob/master/docs/langchain-retrieval-agent.ipynb)

#### [LangChain Handbook](https://pinecone.io/learn/langchain)

# Retrieval Agents

We've seen in previous chapters how powerful [retrieval augmentation](https://www.pinecone.io/learn/langchain-retrieval-augmentation/) and [conversational agents](https://www.pinecone.io/learn/langchain-agents/) can be. They become even more impressive when we begin using them together.

Conversational agents can struggle with data freshness, knowledge about specific domains, or accessing internal documentation. By coupling agents with retrieval augmentation tools we no longer have these problems.

One the other side, using "naive" retrieval augmentation without the use of an agent means we will retrieve contexts with *every* query. Again, this isn't always ideal as not every query requires access to external knowledge.

Merging these methods gives us the best of both worlds. In this notebook we'll learn how to do this.

[![Open full notebook](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/full-link.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/08-langchain-retrieval-agent.ipynb)

To begin, we must install the prerequisite libraries that we will be using in this notebook.

!pip install -qU \
    openai==0.27.7 \
    pinecone-client==3.1.0 \
    pinecone-datasets==0.7.0 \
    langchain==0.1.1 \
    langchain-community==0.0.13 \
    tiktoken==0.4.0 \
    pinecone-notebooks==0.1.1

## Building the Knowledge Base

We will download a pre-embedded dataset from `pinecone-datasets`. Allowing us to skip the embedding and preprocessing steps, if you'd rather work through those steps you can find the [full notebook here](https://colab.research.google.com/github/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/08-langchain-retrieval-agent.ipynb).

from pinecone_datasets import load_dataset

dataset = load_dataset("squad-text-embedding-ada-002")
dataset.head()

len(dataset)

We'll format the dataset ready for upsert and reduce what we use to a subset of the full dataset.

# we drop sparse_values as they are not needed for this example
dataset.documents.drop(['sparse_values', 'blob'], axis=1, inplace=True)

dataset.head()

## Creating an Index

Now the data is ready, we can set up our index to store it.

We begin by initializing our connection to Pinecone. To do this we need a [free API key](https://app.pinecone.io).

import os

if not os.environ.get("PINECONE_API_KEY"):
    from pinecone_notebooks.colab import Authenticate
    Authenticate()

from pinecone import Pinecone

api_key = os.environ.get("PINECONE_API_KEY")

# configure client
pc = Pinecone(api_key=api_key)

Now we setup our index specification, this allows us to define the cloud provider and region where we want to deploy our index. You can find a list of all [available providers and regions here](https://docs.pinecone.io/docs/projects).

from pinecone import ServerlessSpec

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

index_name = 'langchain-retrieval-agent-fast'

import time

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

# we create a new index
pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric='dotproduct',
        spec=spec
    )

# wait for index to be initialized
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

Then connect to the index:

index = pc.Index(index_name)
index.describe_index_stats()

We should see that the new Pinecone index has a `total_vector_count` of `0`, as we haven't added any vectors yet.

Now we upsert the data to Pinecone:

index.upsert_from_dataframe(dataset.documents, batch_size=100)

We've indexed everything, now we can check the number of vectors in our index like so:

index.describe_index_stats()

## Creating a Vector Store and Querying

from langchain.embeddings.openai import OpenAIEmbeddings

openai_api_key = os.environ.get('OPENAI_API_KEY') or 'OPENAI_API_KEY'
model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openai_api_key
)

Now that we've build our index we can switch back over to LangChain. We start by initializing a vector store using the same index we just built. We do that like so:

from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pc.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

As in previous examples, we can use the `similarity_search` method to do a pure semantic search (without the generation component).

query = "when was the college of engineering in the University of Notre Dame established?"

vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)

Looks like we're getting good results. Let's take a look at how we can begin integrating this into a conversational agent.

## Initializing the Conversational Agent

Our conversational agent needs a Chat LLM, conversational memory, and a `RetrievalQA` chain to initialize. We create these using:

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

Using these we can generate an answer using the `run` method:

qa.run(query)

But this isn't yet ready for our conversational agent. For that we need to convert this retrieval chain into a tool. We do that like so:

from langchain.agents import Tool

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering general knowledge queries to get '
            'more information about the topic'
        )
    )
]

Now we can initialize the agent like so:

from langchain.agents import initialize_agent

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

With that our retrieval augmented conversational agent is ready and we can begin using it.

### Using the Conversational Agent

To make queries we simply call the `agent` directly.

agent(query)

Looks great, now what if we ask it a non-general knowledge question?

agent("what is 2 * 7?")

Perfect, the agent is able to recognize that it doesn't need to refer to it's general knowledge tool for that question. Let's try some more questions.

agent("can you tell me some facts about the University of Notre Dame?")

agent("can you summarize these facts in two short sentences")

Looks great! We're also able to ask questions that refer to previous interactions in the conversation and the agent is able to refer to the conversation history to as a source of information.

That's all for this example of building a retrieval augmented conversational agent with OpenAI and Pinecone (the OP stack) and LangChain.

Once finished, we delete the Pinecone index to save resources:

pc.delete_index(index_name)

---
