import tokenizer

vocab_path = "E:\\IDEAProjects2\\vocab.txt"
tokenizer = tokenizer.Tokenizer(vocab_path,type='java')

data = "public static void main(String[] args){} bla"
text = tokenizer._tokenize(data)
print(text)

tok_text = tokenizer.convert_tokens_to_ids(text)
print(tok_text)
back = tokenizer.convert_ids_to_tokens(tok_text)
print(back)
print(tokenizer.get_vocab_len())
