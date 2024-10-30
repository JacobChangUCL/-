import os
import torch
import math
from transformers import GPT2LMHeadModel

def read_tokenized_data(file_path):
    with open(file_path, 'r') as f:
        tokens = f.read().strip().split()
    tokens = [int(token) for token in tokens ]
    for i,token in enumerate(tokens):
        if i%8==0:
            tokens[i]+=1
    return tokens

def main():
    # 加载模型
    model_path = '../Datasets/model/model_epoch200'
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 读取测试数据
    tokenized_data_path = '../Datasets/tokenized/'
    test_file = os.path.join(tokenized_data_path, 'tokenized_train_9.txt')
    tokens = read_tokenized_data(test_file)

    # 准备数据
    n_ctx = model.config.n_ctx
    stride = 128
    batch_size = 2

    start_point = 0
    samples = []
    while start_point < len(tokens) - n_ctx:
        samples.append(tokens[start_point: start_point + n_ctx])
        start_point += stride
    if start_point < len(tokens):
        samples.append(tokens[len(tokens) - n_ctx:])

    batches = []
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        batch_inputs = [torch.tensor(sample).long() for sample in batch_samples]
        batch_inputs = torch.stack(batch_inputs)
        batches.append(batch_inputs)

    # 计算困惑度
    total_loss = 0.0
    total_length = 0

    with torch.no_grad():
        for batch_inputs in batches:
            batch_inputs = batch_inputs.to(device)
            outputs = model(input_ids=batch_inputs, labels=batch_inputs)
            loss = outputs[0]  # 获取平均损失

            # 计算此批次的总损失
            batch_loss = loss.item() * batch_inputs.numel()
            total_loss += batch_loss
            total_length += batch_inputs.numel()

    # 计算平均损失和困惑度
    average_loss = total_loss / total_length
    perplexity = math.exp(average_loss)
    print(f'Perplexity on {test_file}: {perplexity}')

if __name__ == '__main__':
    main()
