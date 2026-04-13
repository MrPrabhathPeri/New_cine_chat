import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import requests 
import time
from groq import Groq

# --------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------
st.set_page_config(page_title="Cine-Chat", page_icon="🎬")

st.title("🎬 Cine-Chat: The AI Movie Expert")
st.caption("Powered by Llama 3.3 & RAG")

# --------------------------------------------------------------
# SETUP (CACHED)
# --------------------------------------------------------------
@st.cache_resource
def load_resources():
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        TMDB_API_KEY = st.secrets["TMDB_API_KEY"]  
    except:
        GROQ_API_KEY = GROQ_API_KEY
        TMDB_API_KEY = TMDB_API_KEY                

    client = Groq(api_key=GROQ_API_KEY)

    db_path = "movie_db"
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name="movies", embedding_function=sentence_transformer_ef)

    if collection.count() == 0:
        st.info("Building database... (~1 min)")
        if not os.path.exists('tmdb_5000_movies.csv'):
            st.error("CSV file not found. Please upload it!")
            st.stop()

        df = pd.read_csv('tmdb_5000_movies.csv')
        import json
        def extract_names(text):
            try: return " ".join([item['name'] for item in json.loads(text)])
            except: return ""

        df['combined_text'] = ("Genre: " + df['genres'].apply(extract_names) +
                               " Keywords: " + df['keywords'].apply(extract_names) +
                               " Plot: " + df['overview'].fillna(""))

        ids = [str(i) for i in df['id'].tolist()]
        documents = df['combined_text'].tolist()
        metadatas = df[['title', 'id']].to_dict(orient='records') 

        batch_size = 200
        for i in range(0, len(df), batch_size):
            end = min(i + batch_size, len(df))
            collection.add(ids=ids[i:end], documents=documents[i:end], metadatas=metadatas[i:end])
        st.success("Database built!")

    return client, collection, TMDB_API_KEY  

client, collection, TMDB_API_KEY = load_resources()  
st.write("API KEY:", TMDB_API_KEY)

# ← Entire function replaced with TMDB version
def get_poster(movie_id):
    for attempt in range(6):
        try:
            url = (
                f"https://api.themoviedb.org/3/movie/{movie_id}"
                f"?api_key={TMDB_API_KEY}"
                f"&language=en-US"
            )
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            else:
                return "https://via.placeholder.com/300x450?text=No+Poster"
        except requests.exceptions.RequestException as e:
            print(f"[Attempt {attempt + 1}] Failed for ID {movie_id}: {e}")
            time.sleep(0.3)
    return None


# --------------------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am a movie expert. Ask me anything!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask for a movie recommendation..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    results = collection.query(query_texts=[prompt], n_results=3)

    context_text = ""
    for i, doc in enumerate(results['documents'][0]):
        title = results['metadatas'][0][i]['title']
        context_text += f"Movie: {title}\nPlot: {doc}\n\n"

    system_prompt = f"""
    You are a movie expert. The user wants a recommendation.
    Here are 3 relevant movies:
    {context_text}
    Recommend the best one and explain why.
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )

    response = chat_completion.choices[0].message.content
    st.chat_message("assistant").write(response)
    cols = st.columns(3)
    for i, meta in enumerate(results['metadatas'][0]):
        st.write(meta)
        poster = get_poster(meta['id'])  # ← changed from meta['title'] to meta['id']
        with cols[i]:
            if poster:
                st.image(poster, caption=meta['title'], use_column_width=True)
            else:
                st.caption(f"🎬 {meta['title']}")

    st.session_state.messages.append({"role": "assistant", "content": response})
