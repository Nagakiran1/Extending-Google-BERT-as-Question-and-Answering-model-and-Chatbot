# Extending Google-BERT as Question and Answering model and Chatbot

## BERT Introduction 


BERT( Bidirectional Encoder Representations from Transforers) method of pre-training language representations. With the use of pre-trained BERT models we can utilize a pre-trained memory information of sentence structure, language and text grammar related memory of large corpus of millions, or billions, of annotated training examples that it has trained.




How BERT works
BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. In its vanilla form, Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task. Since BERT’s goal is to generate a language model, only the encoder mechanism is necessary. The detailed workings of Transformer are described in a paper by Google.
 

The pre-trained model can then be fine-tuned on small-data NLP tasks like question answering and sentiment analysis, resulting in substantial accuracy improvements compared to training on these datasets from scratch. 
•	BERT is a huge model, with 24 Transformer blocks, 1024 hidden units in each layer, and 340M parameters.
•	The model is pre-trained on 40 epochs over a 3.3 billion word corpus, including BooksCorpus (800 million words) and English Wikipedia (2.5 billion words).

For more information on BERT , we can go through the [BERT Google AI blog ](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)






BERT uses character-based tokenization for Chinese, and WordPiece tokenization for
all other languages. Both models should work out-of-the-box without any code
changes. We did update the implementation of `BasicTokenizer` in
`tokenization.py` to support Chinese character tokenization, so please update if
you forked it. However, we did not change the tokenization API.

For more, see the
[Multilingual README](https://github.com/google-research/bert/blob/master/multilingual.md).


**BERT**, or **B**idirectional **E**ncoder **R**epresentations from
**T**ransformers, is a new method of pre-training language representations which
obtains state-of-the-art results on a wide array of Natural Language Processing
(NLP) tasks.

To give a few numbers, here are the results on the
[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) question answering
task:





# Using BERT for Question and Answering


Bert model is well defined in understanding the given Text summary and answering the question from that summary. To understand the Question related information Bert has trained on SQUAD data set and other labeled Question and answer dataset .
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.


SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 new, unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. 


Because SQuAD is an ongoing effort, Its not exposed to public as open source dataset, sample data can be downloaded from [SQUAD site ](https://rajpurkar.github.io/SQuAD-explorer/)

To get the whole data set, individual model has to be build in understanding the Summary and to answer the question from sample train dataset and has to be submitted, based on the evaluation criteria, data set will be shared.
Here in BERT model with the use of run_squad.py file, we can get the answer for specific question for the specific sample train data given in SQuAD. To make it to work for individual Summary code has edited and given,
You can try of executing the run_squad.py file for extracting the Answer from based on summary understanding.

### SQuAD 1.1

The Stanford Question Answering Dataset (SQuAD) is a popular question answering
benchmark dataset. BERT (at the time of the release) obtains state-of-the-art
results on SQuAD with almost no task-specific network architecture modifications
or data augmentation. However, it does require semi-complex data pre-processing
and post-processing to deal with (a) the variable-length nature of SQuAD context
paragraphs, and (b) the character-level answer annotations which are used for
SQuAD training. This processing is implemented and documented in `run_squad.py`.




To run on SQuAD, you will first need to download the dataset. The
[SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) does not seem to
link to the v1.1 datasets any longer, but the necessary files can be found here:

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

Download these to some directory `$SQUAD_DIR`.

The state-of-the-art SQuAD results from the paper currently cannot be reproduced
on a 12GB-16GB GPU due to memory constraints (in fact, even batch size 1 does
not seem to fit on a 12GB GPU using `BERT-Large`). However, a reasonably strong
`BERT-Base` model can be trained on the GPU with these hyperparameters:


![alt text](https://github.com/Nagakiran1/Extending-Google-BERT-as-Question-and-Answering-model-and-Chatbot/blob/master/BERT_Qn_Ans.png)



#### Sample JSON file information used in SQuAD data set, 
```
{'data': [{'title': 'Super_Bowl_50',
   'paragraphs': [{'context': 'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.',

     	'qas': [{'answers': [{'answer_start': 177, 'text': 'Denver Broncos'},
        	{'answer_start': 177, 'text': 'Denver Broncos'},
        	{'answer_start': 177, 'text': 'Denver Broncos'}],
       		'question': 'Which NFL team represented the AFC at Super Bowl 50?',
       		'id': '56be4db0acb8001400a502ec'}]}]}],
 		'version': '1.1'}
```


Squad data comprises with Text summary of some information and the related questions with answer Start and answer string information with in the paragraph summary given.


To utilize the BERT model for the Question and answering, there is a need of preparing the data similar to the SQuAD data structure mentioned.


Text information should be mentioned to the context key of JSON format given
Features used by the BERT model for understanding the given Context




High-level description of the Transformer encoder. The input is a sequence of tokens, which embedded into vectors and then processed in the neural network. The output is a sequence of vectors of size H, in which each vector corresponds to an input token with the same index.

Steps to use the model BERT for Question and answer from a summary 

## Command to Understand the unknown context with BERT and answering the asked Question
Run run_squad.py with mentioning arguments of context and question for question and answering purpose
here 'Context' variable represents the Unknown string of data and 'question' is the asked question from that context.
```
python run_squad.py \
  --vocab_file=BERT_BASE_DIR/vocab.txt \
  --bert_config_file=BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=BERT_BASE_DIR/bert_model.ckpt \
  --do_predict=True\
  --context= Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.\
  --output_dir=Data
  --question=Which NFL team represented the AFC at Super Bowl 50?
```

 

#### Features Generated for the Sample JSON data context shown above

For more information about BERT features explanation please refer [BERT Embeddings Tutorial] (https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)


1)	Token to orig map :
Consideering the index of each word as a feature integer
```
INFO:tensorflow:token_to_orig_map: 17:0 18:1 19:2 20:3 21:4 22:5 23:5 24:5 25:6 26:7 27:8 28:9 29:10 30:11 31:12 32:13 33:14 34:15 35:16 36:17 37:17 38:17 39:17 40:17 41:18 42:19 43:20 44:21 45:21 46:22 47:23 48:23 49:23 50:24 51:25 52:26 53:26 54:26 55:26 56:26 57:27 58:28 59:28 60:29 61:29 62:29 63:30 64:31 65:32 66:33 67:34 68:35 69:35 70:35 71:35 72:35 73:36 74:37 75:37 76:37 77:38 78:38 79:38 80:39 81:39 82:39 83:40 84:41 85:42 86:43 87:44 88:45 89:46 90:46 91:47 92:48 93:49 94:50 95:51 96:52 97:52 98:52 99:52 100:53 101:53 102:54 103:54 104:55 105:56 106:56 107:56 108:56 109:57 110:58 111:59 112:60 113:60 114:61 115:61 116:61 117:61 118:62 119:63 120:64 121:65 122:65 123:66 124:66 125:66 126:66 127:67 128:67 129:67 130:67 131:67 132:67 133:68 134:69 135:70 136:71 137:72 138:73 139:74 140:74 141:75 142:76 143:77 144:78 145:79 146:79 147:80 148:80 149:81 150:82 151:83 152:83 153:83 154:84 155:84 156:85 157:86 158:87 159:88 160:89 161:89 162:89 163:90 164:91 165:92 166:93 167:94 168:95 169:96 170:97 171:98 172:99 173:99 174:100 175:100 176:100 177:101 178:101 179:102 180:103 181:104 182:105 183:106 184:107 185:108 186:109 187:110 188:110 189:111 190:112 191:112 192:112 193:112 194:113 195:114 196:115 197:116 198:117 199:118 200:119 201:120 202:121 203:121 204:121 205:122 206:122 207:122 208:123 209:123

```
2)	Token is max content :
A Boolean index to each word representing whether the word is important in the context or not.

```
INFO:tensorflow:token_is_max_context: 17:True 18:True 19:True 20:True 21:True 22:True 23:True 24:True 25:True 26:True 27:True 28:True 29:True 30:True 31:True 32:True 33:True 34:True 35:True 36:True 37:True 38:True 39:True 40:True 41:True 42:True 43:True 44:True 45:True 46:True 47:True 48:True 49:True 50:True 51:True 52:True 53:True 54:True 55:True 56:True 57:True 58:True 59:True 60:True 61:True 62:True 63:True 64:True 65:True 66:True 67:True 68:True 6	9:True 70:True 71:True 72:True 73:True 74:True 75:True 76:True 77:True 78:True 79:True 80:True 81:True 82:True 83:True 84:True 85:True 86:True 87:True 88:True 89:True 90:True 91:True 92:True 93:True 94:True 95:True 96:True 97:True 98:True 99:True 100:True 101:True 102:True 103:True 104:True 105:True 106:True 107:True 108:True 109:True 110:True 111:True 112:True 113:True 114:True 115:True 116:True 117:True 118:True 119:True 120:True 121:True 122:True 123:True 124:True 125:True 126:True 127:True 128:True 129:True 130:True 131:True 132:True 133:True 134:True 135:True 136:True 137:True 138:True 139:True 140:True 141:True 142:True 143:True 144:True 145:True 146:True 147:True 148:True 149:True 150:True 151:True 152:True 153:True 154:True 155:True 156:True 157:True 158:True 159:True 160:True 161:True 162:True 163:True 164:True 165:True 166:True 167:True 168:True 169:True 170:True 171:True 172:True 173:True 174:True 175:True 176:True 177:True 178:True 179:True 180:True 181:True 182:True 183:True 184:True 185:True 186:True 187:True 188:True 189:True 190:True 191:True 192:True 193:True 194:True 195:True 196:True 197:True 198:True 199:True 200:True 201:True 202:True 203:True 204:True 205:True 206:True 207:True 208:True 209:True

```

3)	Input_ids
Assigning word to vector ids to each word of context by considering the words dictionary data from pretrained BERT.

```
INFO:tensorflow:input_ids: 101 1134 183 2087 1233 1264 2533 1103 170 2087 1665 1120 7688 7329 1851 136 102 7688 7329 1851 1108 1126 1821 26237 1389 1709 1342 1106 4959 1103 3628 1104 1103 1569 1709 2074 113 183 2087 1233 114 1111 1103 1410 1265 119 1103 1821 26237 1389 1709 3511 113 170 2087 1665 114 3628 10552 4121 9304 1320 13538 2378 1103 1569 1709 3511 113 183 2087 1665 114 3628 1610 27719 1161 13316 8420 1116 1572 782 1275 1106 7379 1147 1503 7688 7329 1641 119 1103 1342 1108 1307 1113 175 15581 5082 3113 128 117 1446 117 1120 5837 5086 112 188 4706 1107 1103 21718 1179 175 4047 21349 2528 5952 1298 1120 21718 13130 172 5815 1161 117 11019 2646 14467 4558 1465 119 1112 1142 1108 1103 13163 7688 7329 117 1103 2074 13463 1103 107 5404 5453 107 1114 1672 2284 118 12005 11751 117 1112 1218 1112 7818 28117 20080 16264 1103 3904 1104 10505 1296 7688 7329 1342 1114 187 27085 183 15447 16179 113 1223 1134 1103 1342 1156 1138 1151 1227 1112 107 7688 7329 181 107 114 117 1177 1115 1103 7998 1180 15199 2672 1103 170 17952 1596 183 15447 16179 1851 119 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

```

4)	Input_mask :
Boolean mask to the context words based on their presence and followed by zero padding to meet the max word vector representation used for specific downloaded BERT model.

```
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0

```

5)	Segment_ids 
segment ids to indicate whether a token belongs to the first sequence or the second sequence. - generate valid length 	 	 

```
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

```

Which should produce an output like this:
Output for the Question asked from the summary

```shell
American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference
```
The Context passed in command and the question mentioned in the command can be changed according the requirement. 
BERT able to understands the context by learning through the Features generated and will be able to answer the question to the extent of trained data.


## Extending BERT as Chatbot for Specific data

BERT Question and Answer system meant and works well for only limited number of words summary like 1 to 2 paragraphs only. It can’t be able to answer well from understanding more than 10 pages of data. We can’t expect to have good results with large corpus as it is trained on SQuAD question and answer data, which has less summary paragraph and questions related to that.


We can extend the BERT question and answer model to work as chatbot on large text. To accomplish the understanding of more than 10 pages of data, here we have used a specific approach of picking the data.  	

```
DocumentData.txt

The pre-trained model can then be fine-tuned on small-data NLP tasks like question answering and sentiment analysis, resulting in substantial accuracy improvements compared to training on these datasets from scratch. 


This week, we open sourced a new technique for NLP pre-training called Bidirectional Encoder Representations from Transformers, or BERT. With this release, anyone in the world can train their own state-of-the-art question answering system (or a variety of other models) in about 30 minutes on a single Cloud TPU, or in a few hours using a single GPU. 


What Makes BERT Different?
BERT builds upon recent work in pre-training contextual representations — including Semi-supervised Sequence Learning, Generative Pre-Training, ELMo, and ULMFit. However, unlike these previous models, BERT is the first deeply bidirectional, unsupervised language representation, pre-trained using only a plain text corpus (in this case, Wikipedia).
	

The Strength of Bidirectionality
If bidirectionality is so powerful, why hasn’t it been done before? To understand why, consider that unidirectional models are efficiently trained by predicting each word conditioned on the previous words in the sentence
```
#### Data Preparation 
The data wants to for the Questioning the BERT has to be copied in simple text file, delimitting with three '\n'.

Data should be with different clusters of deviation information to use similarity algorithm well.

![alt text](https://github.com/Nagakiran1/Extending-Google-BERT-as-Question-and-Answering-model-and-Chatbot/blob/master/BERT_as_chatbot.png)

Here, We first Segregate the data into parts of paragraphs while passing it to model. Once if the data is segregated, based on the question asked we will get the most similar paragraphs containing answer related information using word_vectors of gensim model. 
Data Preparation to feed the

Data preparation is very simple, here we are intended to feed the most similar paragraph to BERT, than to feed all the document of data that we have in questioning.



```
python run_squad.py \
  --vocab_file=BERT_BASE_DIR/vocab.txt \
  --bert_config_file=BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=BERT_BASE_DIR/bert_model.ckpt \
  --do_predict=True\
  --interact=True\
  --context=Data/data.txt \
  --output_dir=Data
```	



Which should produce an interactive chatbot interface like this:
Output for the Question asked from the summary

```shell
>>>'what are bidirectional usage 	of BERT‘
consider that unidirectional models 	are efficiently trained by predicting 	each word conditioned on the 	previous words in the sentence
>>>what are the uses of pre-trained 	models
fine-tuned on small-data NLP tasks like question answering and sentiment analysis
>>>quit
```


#### The added Code does'nt change the basic functionalities BERT meant for, We can use the same Configurations for the training and predictions.
#### The following command allows to train the model with the train sample SQuAD data and makes Predictions for the Dev set data.


```shell
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
```




The dev set predictions will be saved into a file called `predictions.json` in
the `output_dir`:

```shell
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ./squad/predictions.json
```

Which should produce an output like this:

```shell
{"f1": 88.41249612335034, "exact_match": 81.2488174077578}
```

You should see a result similar to the 88.5% reported in the paper for
`BERT-Base`.

If you have access to a Cloud TPU, you can train with `BERT-Large`. Here is a
set of hyperparameters (slightly different than the paper) which consistently
obtain around 90.5%-91.0% F1 single-system trained only on SQuAD:

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME
```

For example, one random run with these parameters produces the following Dev
scores:

```shell
{"f1": 90.87081895814865, "exact_match": 84.38978240302744}
```

If you fine-tune for one epoch on
[TriviaQA](http://nlp.cs.washington.edu/triviaqa/) before this the results will
be even better, but you will need to convert TriviaQA into the SQuAD json
format.

### SQuAD 2.0

This model is also implemented and documented in `run_squad.py`.

To run on SQuAD 2.0, you will first need to download the dataset. The necessary
files can be found here:

*   [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
*   [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
*   [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

Download these to some directory `$SQUAD_DIR`.

On Cloud TPU you can run with BERT-Large as follows:

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True
```

We assume you have copied everything from the output directory to a local
directory called ./squad/. The initial dev set predictions will be at
./squad/predictions.json and the differences between the score of no answer ("")
and the best non-null answer for each question will be in the file
./squad/null_odds.json

Run this script to tune a threshold for predicting null versus non-null answers:

python $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/dev-v2.0.json
./squad/predictions.json --na-prob-file ./squad/null_odds.json

Assume the script outputs "best_f1_thresh" THRESH. (Typical values are between
-1.0 and -5.0). You can now re-run the model to generate predictions with the
derived threshold or alternatively you can extract the appropriate answers from
./squad/nbest_predictions.json.

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True \
  --null_score_diff_threshold=$THRESH
```








SQuAD v1.1 Leaderboard (Oct 8th 2018) | Test EM  | Test F1
------------------------------------- | :------: | :------:
1st Place Ensemble - BERT             | **87.4** | **93.2**
2nd Place Ensemble - nlnet            | 86.0     | 91.7
1st Place Single Model - BERT         | **85.1** | **91.8**
2nd Place Single Model - nlnet        | 83.5     | 90.1

And several natural language inference tasks:

System                  | MultiNLI | Question NLI | SWAG
----------------------- | :------: | :----------: | :------:
BERT                    | **86.7** | **91.1**     | **86.3**
OpenAI GPT (Prev. SOTA) | 82.2     | 88.1         | 75.0

Plus many other tasks.

Moreover, these results were all obtained with almost no task-specific neural
network architecture design.

If you already know what BERT is and you just want to get started, you can
[download the pre-trained models](#pre-trained-models) and
[run a state-of-the-art fine-tuning](#fine-tuning-with-bert) in only a few
minutes.

### What is BERT?

BERT is a method of pre-training language representations, meaning that we train
a general-purpose "language understanding" model on a large text corpus (like
Wikipedia), and then use that model for downstream NLP tasks that we care about
(like question answering). BERT outperforms previous methods because it is the
first *unsupervised*, *deeply bidirectional* system for pre-training NLP.

*Unsupervised* means that BERT was trained using only a plain text corpus, which
is important because an enormous amount of plain text data is publicly available
on the web in many languages.
In order to learn relationships between sentences, we also train on a simple
task which can be generated from any monolingual corpus: Given two sentences `A`
and `B`, is `B` the actual next sentence that comes after `A`, or just a random
sentence from the corpus?



For information about the Multilingual and Chinese model, see the
[Multilingual README](https://github.com/google-research/bert/blob/master/multilingual.md).

**When using a cased model, make sure to pass `--do_lower=False` to the training
scripts. (Or pass `do_lower_case=False` directly to `FullTokenizer` if you're
using your own script.)**

The links to the models are here (right-click, 'Save link as...' on the name):

*   **[`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)**:
    12-layer, 768-hidden, 12-heads, 110M parameters
*   **[`BERT-Large, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Base, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)**:
    12-layer, 768-hidden, 12-heads , 110M parameters
*   **[`BERT-Large, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Base, Multilingual Cased (New, recommended)`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)**:
    104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
*   **[`BERT-Base, Multilingual Uncased (Orig, not recommended)`](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip)
    (Not recommended, use `Multilingual Cased` instead)**: 102 languages,
    12-layer, 768-hidden, 12-heads, 110M parameters
*   **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**:
    Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M
    parameters

Each .zip file contains three items:

*   A TensorFlow checkpoint (`bert_model.ckpt`) containing the pre-trained
    weights (which is actually 3 files).
*   A vocab file (`vocab.txt`) to map WordPiece to word id.
*   A config file (`bert_config.json`) which specifies the hyperparameters of
    the model.

## Fine-tuning with BERT

**Important**: All results on the paper were fine-tuned on a single Cloud TPU,
which has 64GB of RAM. It is currently not possible to re-produce most of the
`BERT-Large` results on the paper using a GPU with 12GB - 16GB of RAM, because
the maximum batch size that can fit in memory is too small. We are working on
adding code to this repository which allows for much larger effective batch size
on the GPU. See the section on [out-of-memory issues](#out-of-memory-issues) for
more details.

The fine-tuning examples which use `BERT-Base` should be able to run on a GPU
that has at least 12GB of RAM using the hyperparameters given.



```
  --output_dir=gs://some_bucket/my_output_dir/
```

The unzipped pre-trained model files can also be found in the Google Cloud
Storage folder `gs://bert_models/2018_10_18`. For example:

```
export BERT_BASE_DIR=gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12
```

### Out-of-memory issues

All experiments in the paper were fine-tuned on a Cloud TPU, which has 64GB of
device RAM. Therefore, when using a GPU with 12GB - 16GB of RAM, you are likely
to encounter out-of-memory issues if you use the same hyperparameters described
in the paper.

The factors that affect memory usage are:

*   **`max_seq_length`**: The released models were trained with sequence lengths
    up to 512, but you can fine-tune with a shorter max sequence length to save
    substantial memory. This is controlled by the `max_seq_length` flag in our
    example code.

*   **`train_batch_size`**: The memory usage is also directly proportional to
    the batch size.

*   **Model type, `BERT-Base` vs. `BERT-Large`**: The `BERT-Large` model
    requires significantly more memory than `BERT-Base`.

*   **Optimizer**: The default optimizer for BERT is Adam, which requires a lot
    of extra memory to store the `m` and `v` vectors. Switching to a more memory
    efficient optimizer can reduce memory usage, but can also affect the
    results. We have not experimented with other optimizers for fine-tuning.

Using the default training scripts (`run_classifier.py` and `run_squad.py`), we
benchmarked the maximum batch size on single Titan X GPU (12GB RAM) with
TensorFlow 1.11.0:

System       | Seq Length | Max Batch Size
------------ | ---------- | --------------
`BERT-Base`  | 64         | 64
...          | 128        | 32
...          | 256        | 16
...          | 320        | 14
...          | 384        | 12
...          | 512        | 6
`BERT-Large` | 64         | 12
...          | 128        | 6
...          | 256        | 2
...          | 320        | 1
...          | 384        | 0
...          | 512        | 0


