from A import tokenization_bert_chinese

vocab_file_path = 'Datasets/vocab_file.txt'

def main():
    tokenizer = tokenization_bert_chinese.BertTokenizer(vocab_file=vocab_file_path)

    tokens1 = tokenizer.tokenize("务民之义敬鬼神而远之可谓知矣")
    print(tokens1)
    context_tokens1 = tokenizer.convert_tokens_to_ids(tokens1)
    print(context_tokens1)

    tokens2 = tokenizer.tokenize("何为益者三友")
    print(tokens2)
    context_tokens2 = tokenizer.convert_tokens_to_ids(tokens2)
    print(context_tokens2)
   

if __name__ == '__main__':
    main()
