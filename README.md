# Extending Google-BERT as Question and Answering model and Chatbot

# BERT Introduction 


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


## Features used by BERT
![alt text](https://github.com/Nagakiran1/Extending-Google-BERT-as-Question-and-Answering-model-and-Chatbot/blob/master/BERT_Qn_Ans.png)



Sample JSON file information used in SQuAD data set
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

## What is BERT?

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

This code was tested with TensorFlow 1.11.0. It was tested with Python2 and
Python3 (but more thoroughly with Python2, since this is what's used internally
in Google).

The fine-tuning examples which use `BERT-Base` should be able to run on a GPU
that has at least 12GB of RAM using the hyperparameters given.


Please see the
[Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist)
for how to use Cloud TPUs. Alternatively, you can use the Google Colab notebook
"[BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)".

On Cloud TPUs, the pretrained model and the output directory will need to be on
Google Cloud Storage. For example, if you have a bucket named `some_bucket`, you
might use the following flags instead:

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

Unfortunately, these max batch sizes for `BERT-Large` are so small that they
will actually harm the model accuracy, regardless of the learning rate used. We
are working on adding code to this repository which will allow much larger
effective batch sizes to be used on the GPU. The code will be based on one (or
both) of the following techniques:

*   **Gradient accumulation**: The samples in a minibatch are typically
    independent with respect to gradient computation (excluding batch
    normalization, which is not used here). This means that the gradients of
    multiple smaller minibatches can be accumulated before performing the weight
    update, and this will be exactly equivalent to a single larger update.

*   [**Gradient checkpointing**](https://github.com/openai/gradient-checkpointing):
    The major use of GPU/TPU memory during DNN training is caching the
    intermediate activations in the forward pass that are necessary for
    efficient computation in the backward pass. "Gradient checkpointing" trades
    memory for compute time by re-computing the activations in an intelligent
    way.

**However, this is not implemented in the current release.**

### Pre-training tips and caveats

*   **If using your own vocabulary, make sure to change `vocab_size` in
    `bert_config.json`. If you use a larger vocabulary without changing this,
    you will likely get NaNs when training on GPU or TPU due to unchecked
    out-of-bounds access.**
*   If your task has a large domain-specific corpus available (e.g., "movie
    reviews" or "scientific papers"), it will likely be beneficial to run
    additional steps of pre-training on your corpus, starting from the BERT
    checkpoint.
*   The learning rate we used in the paper was 1e-4. However, if you are doing
    additional steps of pre-training starting from an existing BERT checkpoint,
    you should use a smaller learning rate (e.g., 2e-5).
*   Current BERT models are English-only, but we do plan to release a
    multilingual model which has been pre-trained on a lot of languages in the
    near future (hopefully by the end of November 2018).
*   Longer sequences are disproportionately expensive because attention is
    quadratic to the sequence length. In other words, a batch of 64 sequences of
    length 512 is much more expensive than a batch of 256 sequences of
    length 128. The fully-connected/convolutional cost is the same, but the
    attention cost is far greater for the 512-length sequences. Therefore, one
    good recipe is to pre-train for, say, 90,000 steps with a sequence length of
    128 and then for 10,000 additional steps with a sequence length of 512. The
    very long sequences are mostly needed to learn positional embeddings, which
    can be learned fairly quickly. Note that this does require generating the
    data twice with different values of `max_seq_length`.
*   If you are pre-training from scratch, be prepared that pre-training is
    computationally expensive, especially on GPUs. If you are pre-training from
    scratch, our recommended recipe is to pre-train a `BERT-Base` on a single
    [preemptible Cloud TPU v2](https://cloud.google.com/tpu/docs/pricing), which
    takes about 2 weeks at a cost of about $500 USD (based on the pricing in
    October 2018). You will have to scale down the batch size when only training
    on a single Cloud TPU, compared to what was used in the paper. It is
    recommended to use the largest batch size that fits into TPU memory.

### Pre-training data

We will **not** be able to release the pre-processed datasets used in the paper.
For Wikipedia, the recommended pre-processing is to download
[the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2),
extract the text with
[`WikiExtractor.py`](https://github.com/attardi/wikiextractor), and then apply
any necessary cleanup to convert it into plain text.

Unfortunately the researchers who collected the
[BookCorpus](http://yknzhu.wixsite.com/mbweb) no longer have it available for
public download. The
[Project Guttenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html)
is a somewhat smaller (200M word) collection of older books that are public
domain.

[Common Crawl](http://commoncrawl.org/) is another very large collection of
text, but you will likely have to do substantial pre-processing and cleanup to
extract a usable corpus for pre-training BERT.

### Learning a new WordPiece vocabulary

This repository does not include code for *learning* a new WordPiece vocabulary.
The reason is that the code used in the paper was implemented in C++ with
dependencies on Google's internal libraries. For English, it is almost always
better to just start with our vocabulary and pre-trained models. For learning
vocabularies of other languages, there are a number of open source options
available. However, keep in mind that these are not compatible with our
`tokenization.py` library:

*   [Google's SentencePiece library](https://github.com/google/sentencepiece)

*   [tensor2tensor's WordPiece generation script](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder_build_subword.py)

*   [Rico Sennrich's Byte Pair Encoding library](https://github.com/rsennrich/subword-nmt)

## Using BERT in Colab

If you want to use BERT with [Colab](https://colab.research.google.com), you can
get started with the notebook
"[BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)".
**At the time of this writing (October 31st, 2018), Colab users can access a
Cloud TPU completely for free.** Note: One per user, availability limited,
requires a Google Cloud Platform account with storage (although storage may be
purchased with free credit for signing up with GCP), and this capability may not
longer be available in the future. Click on the BERT Colab that was just linked
for more information.

