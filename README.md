The README file should contain 
- a brief description of the organization of your project; 
- the role of each file; 
- the packages required to run your code (e.g. numpy, scipy, transformers, etc.).

# ELEC0141: Deep Learning for Natural Language Processing Assignment 
## Organization of the project
folder A contains the code for tokenization_bert, the role of each file is as follows:
- make_vocab.py: Generate a vocabulary file from the specified original training corpus.
- tokenization_bert_chinese.py: This uses a greedy longest-match-first algorithm to perform tokenization using the given Chinese vocabulary.


## packages required to run your code
- thulac
- jieba
- json
- keras

The following command should be run to generate the vocabulary file:
```
cd A
python make_vocab.py --raw_data_path='../Datasets/train_lunyu.json' --vocab_file_path='../Datasets/vocab_file.txt'

```
- '--raw_data_path' parameter is the original training corpus file path
- '--vocab_file_path' parameter is the path of the generated vocabulary file

The entrance of the project is main.py, which can be run directly in the terminal.
```
python main.py
```

## v1
- 搜集原始训练语料，整理为'Datasets/train_lunyu.json'，包含512行论语句子。句子中所有的标点符号都为中文全角标点。
- 使用'make_vocab.py'生成词表文件'Datasets/vocab_file.txt'
- 编写中文词表tokenization文件'A/tokenization_bert_chinese.py',实现打开词表文件，tokenize指定句子等功能。
- 编写'main.py'文件，使用2个句子测试tokenization_bert_chinese的功能，从结果可以看出tokenization_bert_chinese基本实现了对句子的tokenize功能。
- 将tokenization_bert_chinese中的分词模块换成了jieba,使得分词效果更好。
- 由于jieba分词的结果中可能会出现词表中没有的词，这样会导致这个词的id会用[UNK]代替，因此将这个词拆分为单个字，从而避免了[UNK]的出现。例如'之义'会变成'之'和'义'两个字，'知矣'会变成'知'和'矣'两个字。
```
$ python main.py
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 0.796 seconds.
Prefix dict has been built successfully.
_tokenize_chinese_chars_jieba():
务民 之义 敬鬼神而远之 可谓 知矣
whitespace_tokenize():
['务民', '之义', '敬鬼神而远之', '可谓', '知矣']
['务民', '之', '义', '敬鬼神而远之', '可谓', '知', '矣']
[1270, 13, 161, 1271, 79, 40, 18]
_tokenize_chinese_chars_jieba():
何为 益者 三友
whitespace_tokenize():
['何为', '益者', '三友']
['何', '为', '益者', '三友']
[69, 22, 454, 758]
```

## v2
- 下载中文文言文预训练模型，地址为：https://drive.google.com/drive/folders/1dtHTRn3fX7g8cPCCaJEXA2tmrIcImR6t
- 重命名配置文件config.json为config_lunyu.json,并调整相应参数值。
- 使用argparse添加main.py的运行参数，所有参数都指定默认值。
- 在main.py中添加transformers.modeling_gpt2模块。
- 测试模型加载过程。

## v3
- 加载tokenized_data。
- 配置optimizer和scheduler。
- 训练tokenized_data，log to tensorboard, 保存模型文件。