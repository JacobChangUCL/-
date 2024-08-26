import os
from A import tokenization_bert_chinese
from torch.utils.tensorboard import SummaryWriter
import argparse
import transformers
import torch
import torch.nn.functional as F

def main():
    parser_train = argparse.ArgumentParser()
    parser_train.add_argument('--device', default='0,1', type=str, required=False, help='GPU devices')
    parser_train.add_argument('--model_config', default='Datasets/config_lunyu.json', type=str, required=False, help='model config file')
    parser_train.add_argument('--tokenizer_path', default='Datasets/vocab_file.txt', type=str, required=False, help='vocab file')
    parser_train.add_argument('--raw_data_path', default='Datasets/train_lunyu.json', type=str, required=False, help='raw trainning data')
    parser_train.add_argument('--tokenized_data_path', default='Datasets/tokenized/', type=str, required=False, help='path of tokenized data')
    parser_train.add_argument('--raw', default=True, action='store_true', help='do tokenize first')
    parser_train.add_argument('--epochs', default=20, type=int, required=False, help='epochs')
    parser_train.add_argument('--batch_size', default=2, type=int, required=False, help='batch size')
    parser_train.add_argument('--lr', default=1.5e-4, type=float, required=False, help='learning rate')
    parser_train.add_argument('--warmup_steps', default=9000, type=int, required=False, help='warm up steps')
    parser_train.add_argument('--log_step', default=1, type=int, required=False, help='log loss steps')
    parser_train.add_argument('--stride', default=512, type=int, required=False, help='stride window')
    parser_train.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='gradient accumulation')
    parser_train.add_argument('--fp16', default=False, action='store_true', help='fp16')
    parser_train.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser_train.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser_train.add_argument('--num_pieces', default=10, type=int, required=False, help='trainning pieces')
    parser_train.add_argument('--min_length', default=1, type=int, required=False, help='min length of data')
    parser_train.add_argument('--output_dir', default='Datasets/model/', type=str, required=False, help='output dir for model')
    parser_train.add_argument('--pretrained_model', default='', type=str, required=False, help='pretrained model')
    parser_train.add_argument('--writer_dir', default='Datasets/tensorboard_summary/', type=str, required=False, help='Tensorboard path')
    parser_train.add_argument('--segment', default=False, action='store_true', help='if need split Chinese')

    parser_train.add_argument('--length', default=-1, type=int, required=False, help='length of generation')
    parser_train.add_argument('--nsamples', default=1, type=int, required=False, help='samples')
    parser_train.add_argument('--temperature', default=1, type=float, required=False, help='temperature')
    parser_train.add_argument('--topk', default=8, type=int, required=False, help='top k')
    parser_train.add_argument('--topp', default=0, type=float, required=False, help='topp')
    parser_train.add_argument('--model_path', default='Datasets', type=str, required=False, help='model path')
    parser_train.add_argument('--prefix', default='LUNYU', type=str, required=False, help='start of words')
    parser_train.add_argument('--no_wordpiece', action='store_true', help='no word piece')
    parser_train.add_argument('--fast_pattern', action='store_true', help='if use fast pattern')
    parser_train.add_argument('--repetition_penalty', default=1.0, type=float, required=False)

    args = parser_train.parse_args()
    print('\tTraining args:\n' + args.__repr__())
   
    print("Training...")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('\tTraining: config file:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    tokenizer = tokenization_bert_chinese.BertTokenizer(vocab_file=args.tokenizer_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\t Training: device:', device)

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw 
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)

    model.train()
    model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('\tTraining: number of parameters: {}'.format(num_parameters))


if __name__ == '__main__':
    main()
