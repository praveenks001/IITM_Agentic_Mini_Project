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

(ii) chunk_document - As mentioned, each document from the corpus will be iterated and chunks with 500 chars with an overlap of 40 characters using "Sliding window chunking" method and assign a ID to each chunk.