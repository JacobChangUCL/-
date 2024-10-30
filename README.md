# Deep Learning for Natural Language Processing Assignment
## Organization of the project
```plaintext
DLNLP_assignment_23-24/
│
├── A/                                  # Auxiliary scripts for vocabulary and evaluation
│   │
│   ├── baseline.py                     # Baseline model comparison
│   ├── evaluation.py                   # Model performance evaluation
│   ├── make_vocab.py                   # Vocabulary generation
│   ├── train.py                        # Model training and generating output for the test set
│   └── tokenization_bert_chinese.py    # Custom tokenizer for Chinese text
│
├── Datasets/                           # Directory for datasets
│   └── Dataset_For_Evaluation/         # Evaluation dataset files
│       ├── Model_Output.json           # Model output for evaluation comparison
│       └── Original.json               # Original data for evaluation
│
├── model/                              # Saved model checkpoints
│   ├── model_epoch8                    # Model state after 8 epochs
│   └── model_epoch200                  # Model state after 200 epochs
│
├── tensorboard_summary/                # Logs for monitoring with TensorBoard
│   ├── events.out.tfevents...          # Multiple event logs for training visualization
│
├── tokenized/                          # Tokenized training data from part0 to part9
│   ├── tokenized_train_0.txt           # Tokenized data file part 0
│   ├── tokenized_train_1.txt           # Tokenized data file part 1
│   └── ...                             # Additional tokenized data files
│
├── config_lunyu.json                   # Configuration file for model parameters
├── train_lunyu.json                    # Training data file (Analects sentences)
├── vocab_file.txt                      # Vocabulary generated from the dataset
│
├── main.py                             # Main script for text generation
└── README.md                           # Project description and usage instructions

```



The following command should be run to generate the vocabulary file:
```
cd A
python make_vocab.py --raw_data_path='../Datasets/train_lunyu.json' --vocab_file_path='../Datasets/vocab_file.txt'

```
The entrance of the project is main.py, which can be run directly in the terminal.
```
python main.py
```
Then you can see the following sentence in the terminal.
```
please makesure you have downloaded the fine-turned model."
Choose an option:
1. Run main function
2. Generate model_output file for test set
```
Packages required to run the code
```
- thulac
- jieba
- json
- keras
- torch
- transformer
- evaluate
- numpy
```
You can download model parameters from the following link: 
https://pan.baidu.com/s/1KvtvQGpmzBjsuzFYzer6mQ 
Extraction code: asdf 