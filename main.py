import os
from A import tokenization_bert_chinese
from torch.utils.tensorboard import SummaryWriter
import argparse
import transformers
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime

trainData = []

def readTokenizedData(file)->list[int]:
    """
    Read tokenized data from file, return a list of tokens
    It is used to read txt file in the "tokenized" folder
    """
    with open(file, 'r') as f:
        line = f.read().strip()
    tokens = line.split()
    tokens = [int(token) for token in tokens]
    return tokens

def buildTrainData(data_path, num_pieces):
    """
    Read multiple tokenized data files and combine them into trainData
    """
    for i in range(num_pieces):
        tokens = readTokenizedData(data_path + 'tokenized_train_{}.txt'.format(i))
        trainData.append(tokens)
    

def main():
    #define the command line arguments
    parser_train = argparse.ArgumentParser()
    parser_train.add_argument('--model_config', default='Datasets/config_lunyu.json', type=str, required=False, help='model config file')
    parser_train.add_argument('--tokenizer_path', default='Datasets/vocab_file.txt', type=str, required=False, help='vocab file')
    parser_train.add_argument('--raw_data_path', default='Datasets/train_lunyu.json', type=str, required=False, help='raw trainning data')
    parser_train.add_argument('--tokenized_data_path', default='Datasets/tokenized/', type=str, required=False, help='path of tokenized data')
    parser_train.add_argument('--epochs', default=20, type=int, required=False, help='epochs')
    parser_train.add_argument('--batch_size', default=2, type=int, required=False, help='batch size')
    parser_train.add_argument('--lr', default=1.5e-4, type=float, required=False, help='learning rate')
    parser_train.add_argument('--warmup_steps', default=9000, type=int, required=False, help='warm up steps')
    parser_train.add_argument('--stride', default=128, type=int, required=False, help='stride window')
    parser_train.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser_train.add_argument('--num_pieces', default=10, type=int, required=False, help='trainning pieces')
    parser_train.add_argument('--output_dir', default='Datasets/model/', type=str, required=False, help='output dir for model')
    parser_train.add_argument('--writer_dir', default='Datasets/tensorboard_summary/', type=str, required=False, help='Tensorboard path')

    args = parser_train.parse_args()
    print('\tTraining args:\n' + args.__repr__())
   
    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('\tTraining: config file:\n' + model_config.to_json_string())
    # get the context length from the model config
    n_ctx = model_config.n_ctx 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\t Training: device:', device)

    tokenized_data_path = args.tokenized_data_path
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    # Learning rate warmup: the learning rate is increased linearly from 0 to lr over the first warmup_steps training steps.
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

    buildTrainData(tokenized_data_path, num_pieces) # num_pieces=10

    for i in range(num_pieces):
        full_len += len(trainData[i])
    
    print(full_len) # 17825,the total number of tokens

    total_steps = int(full_len / stride * epochs / batch_size )
    #total_samples = (full_len / stride) * epochs
    #total_steps = int(total_samples / batch_size)
    print('\tTraining: total steps = {}'.format(total_steps))

    #define optimizer and scheduler
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
                samples.append(tokens[len(tokens)-n_ctx:])
               
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # max_grad_norm=1.0

                #  optimizer step
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                #  log to tensorboard
                tb_writer.add_scalar('loss', loss.item() , step)
                print('\tTraining: Step {} of piece {} , loss {}'.format(
                        step + 1,
                        piece_num,
                        running_loss))
                running_loss = 0
                
            piece_num += 1

        print('\tTraining: saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        print('\tTraining: epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('\tTraining: time for one epoch: {}'.format(then - now))

if __name__ == '__main__':
    main()
