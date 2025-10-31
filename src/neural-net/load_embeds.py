import gensim.downloader
from gensim.models import KeyedVectors

embeds = gensim.downloader.load("glove-wiki-gigaword-50")


try: 
    embeds = KeyedVectors.load("../../data/embeddings/glove_embeddings.data")
    print("Loading from local.")
except (FileNotFoundError, EOFError, ValueError):
    embeds = gensim.downloader.load("glove-wiki-gigaword-50")
    embeds.save("../../data/embeddings/glove_embeddings.data")
    print("Downloading from web.")


