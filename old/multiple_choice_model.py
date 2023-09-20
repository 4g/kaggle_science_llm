from transformers import (AutoTokenizer,
                          BertModel, BertForMultipleChoice, AutoModelForMultipleChoice, MobileBertForMultipleChoice,
                          BertPreTrainedModel, DebertaV2ForMultipleChoice)



model = BertForMultipleChoice.from_pretrained('data/allmodels/model/bge-small-en/')
tokenizer = AutoTokenizer.from_pretrained('data/allmodels/model/bge-small-en/')
