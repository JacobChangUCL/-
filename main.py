import torch
import torch.nn.functional as F
from tqdm import trange
import argparse
from transformers import GPT2LMHeadModel
from A import tokenization_bert_chinese as tokenization_bert
import json
def read_and_split_sentences(json_file_path):
    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化一个列表来存储所有句子
    sentences = []

    # 定义分句的正则表达式模式，匹配中文句子结束符号


    for sentence in data:
        sentences.append(sentence)

    return sentences
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # batch size 1 for now - could be updated for more but the code would be less clear
    assert logits.dim() == 1  
    # Safety check
    top_k = min(top_k, logits.size(-1))  
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def generate(model, context, length, n_ctx, tokenizer, temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    #print(generated)
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(**inputs)  
            next_token_logits = outputs[0][0, -1, :]
            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.tolist()[0]





def main():
    # get arguments from command line.
    parser_gen = argparse.ArgumentParser()
    parser_gen.add_argument('--length', default=20, type=int, required=False, help='length of generation') #fixed length
    parser_gen.add_argument('--nsamples', default=1, type=int, required=False, help='number of samples')
    parser_gen.add_argument('--temperature', default=1, type=float, required=False, help='temperature')
    parser_gen.add_argument('--topk', default=8, type=int, required=False, help='top k')
    parser_gen.add_argument('--topp', default=0, type=float, required=False, help='topp')
    parser_gen.add_argument('--vocab_file', default='Datasets/vocab_file.txt', type=str, required=False, help='vocab file path')
    parser_gen.add_argument('--model_path', default='Datasets/model/model_epoch200', type=str, required=False, help='model path')
    parser_gen.add_argument('--prefix', default='曾子曰：“吾闻诸夫子，', type=str, required=False, help='start of words')
    parser_gen.add_argument('--repetition_penalty', default=1.0, type=float, required=False)

    args = parser_gen.parse_args()
    print('\t args:\n' + args.__repr__())

    # set arguments for generation
    length = args.length
    nsamples = args.nsamples
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty
    device = "cuda" if torch.cuda.is_available() else "cpu"


    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.vocab_file)


    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx

    if length == -1:
        length = model.config.n_ctx

    while True:
        raw_text = args.prefix
        tokenize_raw = tokenizer.tokenize(raw_text)
        context_tokens = tokenizer.convert_tokens_to_ids(tokenize_raw)
        #print(context_tokens)

        generated = 0
        for _ in range(nsamples):
            out = generate(
                model=model,
                context=context_tokens,
                length=length,
                n_ctx=n_ctx,
                tokenizer=tokenizer,
                temperature=temperature, top_k=topk, top_p=topp, repitition_penalty=repetition_penalty,
                device=device
            )
            generated += 1
            text = tokenizer.convert_ids_to_tokens(out)

            for i, item in enumerate(text):
                if item == '[MASK]':
                    text[i] = ''
                elif item == '[CLS]':
                    text[i] = '\n\n'
                elif item == '[SEP]':
                    text[i] = '\n'

            info = "=" * 35 + " SAMPLE " + str(generated) + " " + "=" * 35 + "\n"
            print(info)
            text = ''.join(text).replace('##', '').strip()
            print(text)
        print("=" * 80)
        if generated == nsamples:
            break

def generate_model_output(eval_file="Datasets/Dataset_For_Evaluation/Original.json",first_x=5,generated_file="Datasets/Dataset_For_Evaluation/Model_Output.json"):
    # get arguments from command line.
    parser_gen = argparse.ArgumentParser()

    parser_gen.add_argument('--nsamples', default=1, type=int, required=False, help='number of samples')
    parser_gen.add_argument('--temperature', default=1, type=float, required=False, help='temperature')
    parser_gen.add_argument('--topk', default=8, type=int, required=False, help='top k')
    parser_gen.add_argument('--topp', default=0, type=float, required=False, help='topp')
    parser_gen.add_argument('--vocab_file', default='Datasets/vocab_file.txt', type=str, required=False,
                            help='vocab file path')
    parser_gen.add_argument('--model_path', default='Datasets/model/model_epoch200', type=str, required=False,
                            help='model path')
    parser_gen.add_argument('--prefix', default='曾子曰：“吾闻诸夫子，', type=str, required=False, help='start of words')
    parser_gen.add_argument('--repetition_penalty', default=1.0, type=float, required=False)

    args = parser_gen.parse_args()
    print('\t args:\n' + args.__repr__())

    # set arguments for generation

    nsamples = args.nsamples
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.vocab_file)

    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx


# now we start to generate the output for evaluation
    sentences_list = read_and_split_sentences(eval_file)
    predictions = []
    for j in range(len(sentences_list)):
        print("\nOriginal text {}=".format(j),sentences_list[j])
        raw_text = sentences_list[j][:first_x] #use first_x token as the prefix for generation
        generate_len=max(first_x,len(sentences_list[j])) #generate at least the length of the sentence
        tokenize_raw = tokenizer.tokenize(raw_text)
        context_tokens = tokenizer.convert_tokens_to_ids(tokenize_raw)
        # print(context_tokens)



        generated = 0
        for _ in range(nsamples):
            out = generate(
                model=model,
                context=context_tokens,
                length=generate_len,
                n_ctx=n_ctx,
                tokenizer=tokenizer,
                temperature=temperature, top_k=topk, top_p=topp, repitition_penalty=repetition_penalty,
                device=device
            )
            generated += 1
            text = tokenizer.convert_ids_to_tokens(out)

            for i, item in enumerate(text):
                if item == '[MASK]':
                    text[i] = ''
                elif item == '[CLS]':
                    text[i] = ''
                elif item == '[SEP]':
                    text[i] = ''
            text = ''.join(text).replace('##', '').strip()
            print("Generated text for input {}".format(j),text)
            predictions.append(text)

        # 将列表写入 JSON 文件
        with open('Datasets/Dataset_For_Evaluation/Model_Output.json','w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    while 1:
        print("please makesure you have downloaded the fine-turned model.")
        print("Choose an option:")
        print("1. Run main function")
        print("2. Generate model_output file for test set")

        choice = input("Enter the number of your choice: ")

        if choice == '1':
            main()
            break
        elif choice == '2':
            generate_model_output()
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

