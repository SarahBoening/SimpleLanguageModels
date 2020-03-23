from gensim.models import KeyedVectors as word2vec
import datetime

vectors_text_path = 'H:\\Master\\token_vecs\\token_vecs.txt' # or: `models/java14_model/tokens.txt'
print("loading the model")
then = datetime.datetime.now()
model = word2vec.load_word2vec_format(vectors_text_path, binary=False)
now = datetime.datetime.now()
print("done")
print("elapsed time for prediction: ", now - then)
#print(model.most_similar(positive=['equals', 'tolower'])) # or: 'tolower', if using the downloaded embeddings
#print(model.most_similar(positive=['download', 'send'], negative=['receive']))
wv = model.wv['public']
print("wv: ", wv)
similars = model.wv.most_similar(positive=[wv,])[0]
print(similars)
