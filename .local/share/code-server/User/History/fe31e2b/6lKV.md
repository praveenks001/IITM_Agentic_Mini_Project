Step 1: 

Made the environment ready by changing the Qdrant credentials in .env file
Installed required dependencies for Qdrant

Step 2:

(i) load_corpus -  To consume the Qdrant details from os , we need to load dotenv by adding load_dotenv() along with its import 

from dotenv import load_dotenv
load_dotenv()
assert os.environ.get("QDRANT_URL"),     "Set QDRANT_URL — get free-tier at cloud.qdrant.io"
assert os.environ.get("QDRANT_API_KEY"), "Set QDRANT_API_KEY — from your Qdrant Cloud cluster"

Once this step is done, the qdrant keys are loaded in the environment

(ii) chunk_document - As mentioned in the requirement: In cmd_ingest function, each document from the corpus will be iterated and passed to chunk_document function where in turn it will call another function called "chunk_text" to create chunks of 500 chars with an overlap of 40 characters using "Sliding window chunking" method and assign a ID to each chunk. Like this, all 5 documents are chunked and total chunks are below, 
  01_red_headed_league.txt: 12 chunks
  02_speckled_band.txt: 13 chunks
  03_blue_carbuncle.txt: 12 chunks
  04_engineers_thumb.txt: 14 chunks
  05_scandal_in_bohemia.txt: 16 chunks
→ Total chunks: 67


(iii) setup_collection  - Calling the qdrant collection creation function to create "mp2_sherlock" collection. If its already created, it will just return the collection name and if not, it will create the collection with 1536 dims and return the collection.


(iv) embed_texts  - Using Embbedding model, Each chunk of the corpus will be embedded passing into this method. Create a function called is_qdrant_collection_populated in order to check if the qdrant is available with embedded chunks in it else no need to embed again.


(v) ingest_chunks - if quadrant is already present dont do anything as it can contains embedded chunks already. If not, we can embed the chunks and upsert them into the collection.

(vi) retrieve - Since we are using qdrant, the retrieve function has isQdrant check , if yes - we can retrieve the cosine similarity chunks from qdrant collection