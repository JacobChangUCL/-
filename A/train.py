import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import transformers
import torch
from torch.nn import DataParallel
import numpy as np
import random
from datetime import datetime

trainData = []
testData = []


class EarlyStopping:
    def __init__(self, patience=2, delta=0.1):
        """
        Args:
            patience (int): 在验证集损失不再改善后等待的轮数
            delta (float): 判断改善的阈值
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def readTokenizedData(file):
    with open(file, 'r') as f:
        line = f.read().strip()
    tokens = line.split()
    tokens = [int(token) for token in tokens]
    return tokens


def buildTrainData(data_path, num_pieces):
    for i in range(num_pieces):
        tokens = readTokenizedData(data_path + 'tokenized_train_{}.txt'.format(i))
        trainData.append(tokens)
    # choose the last piece as test data
    tokens = readTokenizedData(data_path + 'tokenized_train_{}.txt'.format(9))
    testData.append(tokens)


def main():
    earlyStopping = EarlyStopping()
    parser_train = argparse.ArgumentParser()
    parser_train.add_argument('--model_config', default='../Datasets/config_lunyu.json', type=str, required=False,
                              help='model config file')
    parser_train.add_argument('--tokenizer_path', default='../Datasets/vocab_file.txt', type=str, required=False,
                              help='vocab file')
    parser_train.add_argument('--raw_data_path', default='../Datasets/train_lunyu.json', type=str, required=False,
                              help='raw trainning data')
    parser_train.add_argument('--tokenized_data_path', default='../Datasets/tokenized/', type=str, required=False,
                              help='path of tokenized data')
    parser_train.add_argument('--epochs', default=200, type=int, required=False, help='epochs')
    parser_train.add_argument('--batch_size', default=2, type=int, required=False, help='batch size')
    parser_train.add_argument('--lr', default=1.5e-4, type=float, required=False, help='learning rate')
    parser_train.add_argument('--warmup_steps', default=9000, type=int, required=False, help='warm up steps')
    parser_train.add_argument('--stride', default=128, type=int, required=False, help='stride window')
    parser_train.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser_train.add_argument('--num_pieces', default=9, type=int, required=False, help='trainning pieces')
    parser_train.add_argument('--output_dir', default='../Datasets/model/', type=str, required=False,
                              help='output dir for model')
    parser_train.add_argument('--writer_dir', default='../Datasets/tensorboard_summary/', type=str, required=False,
                              help='Tensorboard path')

    args = parser_train.parse_args()
    print('\tTraining args:\n' + args.__repr__())

    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('\tTraining: config file:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\t Training: device:', device)

    tokenized_data_path = args.tokenized_data_path
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    stride = args.stride
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)

    print("Training...")

    model.train()
    model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('\tTraining: number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    full_len = 0

    buildTrainData(tokenized_data_path, num_pieces)  # num_pieces=10

    for i in range(num_pieces):
        full_len += len(trainData[i])

    print(full_len)  # 17825

    total_steps = int(full_len / stride * epochs / batch_size)
    print('\tTraining: total steps = {}'.format(total_steps))

    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                             num_training_steps=total_steps)

    if torch.cuda.device_count() > 1:
        print("\tTraining: Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in range(torch.cuda.device_count())])
        multi_gpu = True

    print('\tTraining: starting training')
    overall_step = 0
    running_loss = 0

    global_step = 0  # the global step for tensorboard

    for epoch in range(epochs):
        print('Training: epoch {}'.format(epoch + 1))
        now = datetime.now()
        x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0
        for i in x:
            tokens = trainData[i]
            start_point = 0
            samples = []
            while start_point < len(tokens) - n_ctx:
                samples.append(tokens[start_point: start_point + n_ctx])
                start_point += stride
            if start_point < len(tokens):
                samples.append(tokens[len(tokens) - n_ctx:])

            random.shuffle(samples)

            for step in range(len(samples) // batch_size):  # batch_size=2, drop last

                #  prepare data
                batch = samples[step * batch_size: (step + 1) * batch_size]
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)

                batch_inputs = torch.tensor(batch_inputs).long().to(device)

                #  forward pass
                outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                loss, logits = outputs[:2]

                #  get loss
                if multi_gpu:
                    loss = loss.mean()

                #  loss backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # max_grad_norm=1.0

                #  optimizer step
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                #  log to tensorboard
                tb_writer.add_scalar('loss/train', loss.item(), global_step)
                global_step += 1

                print('\tTraining: Step {} of piece {} , loss {}'.format(
                    step + 1,
                    piece_num,
                    running_loss))
                running_loss = 0

            piece_num += 1

        print('\tTraining: epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('\tTraining: time for one epoch: {}'.format(then - now))

        # 在每个epoch结束后进行测试
        model.eval()  # 切换到评估模式
        with torch.no_grad():
            print('Testing...')
            for tokens in testData:
                start_point = 0
                samples = []
                while start_point < len(tokens) - n_ctx:
                    samples.append(tokens[start_point: start_point + n_ctx])
                    start_point += stride
                if start_point < len(tokens):
                    samples.append(tokens[len(tokens) - n_ctx:])

                random.shuffle(samples)  # 可选，打乱样本顺序

                for step in range(len(samples) // batch_size):
                    # 准备数据
                    batch = samples[step * batch_size: (step + 1) * batch_size]
                    batch_inputs = []
                    for ids in batch:
                        int_ids = [int(x) for x in ids]
                        batch_inputs.append(int_ids)

                    batch_inputs = torch.tensor(batch_inputs).long().to(device)

                    # 前向传播
                    outputs = model(input_ids=batch_inputs, labels=batch_inputs)
                    loss = outputs[0]  # 获取损失

                    # 如果使用多GPU，需要取平均
                    if multi_gpu:
                        loss = loss.mean()

                    # 打印每个批次的损失
                    print('\tTesting: Step {} , loss {}'.format(step + 1, loss.item()))

                    # 记录到TensorBoard

                    tb_writer.add_scalar('loss/test', loss.item(), global_step)

        # early stop
        print("loss.item=", loss.item())
        earlyStopping(loss.item())

        print("earlyStopping.best_loss=", earlyStopping.best_loss)
        print("earlyStopping.counter=", earlyStopping.counter)
        print("earlyStopping.early_stop=", earlyStopping.early_stop)
        # 检查是否需要提前停止
        if earlyStopping.early_stop:
            if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
                os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
            print('\tTraining: saving model for epoch {}'.format(epoch + 1))
            break
        model.train()  # 切换回训练模式


if __name__ == '__main__':
    main()
