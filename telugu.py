!pip3 install transformers
!pip3 install datasets
!pip3 install sentencepiece
!pip3 install seqeval


!pip install transformers

from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-multi-cased-finetuned-xquadv1",
    tokenizer="mrm8488/bert-multi-cased-finetuned-xquadv1"
)

answer=qa_pipeline({
    'context': "రాము అనే వ్యక్తి ఉన్నాడు, అతను విజయవాడలో నివసిస్తున్నాడు మరియు అతని వయస్సు 23 సంవత్సరాలు",
    'question': "రాము ఎక్కడ నివసిస్తున్నాడు?"
    
})
#Exact Match (EM) score: 56.58%
#F1 score: 65%
print(answer)




# Import all the necessary classes and initialize the tokenizer and model.
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")

model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")

#0.93 f1 score




sentence = 'రామ నామము సకల పాప హరమనీ, మోక్షప్రథమనీ పలువురి నమ్మిక. రమంతే సర్వేజనాః గుణైతి ఇతి రామః (తన సద్గుణముల చేత అందరినీ సంతోషింపజేసేవాడు రాముడు) అని రామ శబ్దానికి వ్యుత్పత్తి చెప్పబడింది."రామ" నామములో పంచాక్షరీ మంత్రము "ఓం నమశ్శివాయ" నుండి  బీజాక్షరము, అష్టాక్షరీ మంత్రము "ఓం నమో నారాయణాయ" నుండి  బీజాక్షరము పొందుపరచబడియున్నవని ఆధ్యాత్మిక వేత్తల వివరణ. మూడు మార్లు "రామ" నామమును స్మరించినంతనే శ్రీ విష్ణు సహస్ర నామ స్తోత్రము చేసిన ఫలము లభించునని శ్రీ విష్ణు సహస్ర నామ స్తోత్రము-ఉత్తర పీఠికలో చెప్పబడింది..'

predicted_labels = get_predictions(sentence=sentence, 
                                   tokenizer=tokenizer,
                                   model=model
                                   )

for index in range(len(sentence.split(' '))):
  print( sentence.split(' ')[index] + '\t' + predicted_labels[index] )




  import copy

predictions= get_predictions(sentence=sentence, 
                                   tokenizer=tokenizer,
                                   model=model
                                   )

print(predictions)
print()
p1=[]
predictions=[]



for index in range(len(sentence.split(' '))):
  print( sentence.split(' ')[index] + '\t' + predicted_labels[index] )
  p1.append({sentence.split(' ')[index]:predicted_labels[index]})


predictions.append(p1)
possibleAnswerSpans = []
for i in range(len(predictions)):
  answer = []
  for j in range(len(predictions[i])):
    for key, val in predictions[i][j].items():
      if val.find("-") >= 0:
        answer.append(key)
        # print(answer)
      else:
        if (len(answer) == 0):
          continue
        possibleAnswerSpans.append(copy.deepcopy(answer))
        # print(answer)
        answer = []
        
  if (len(answer) > 0):
    possibleAnswerSpans.append(copy.deepcopy(answer))

print("Possible Answer Spans from given passages: ")
print(possibleAnswerSpans)
print()




from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#The mT5-base-finetuned-tydiQA-question-generation model has been fine-tuned on the TyDiQA dataset, and it has achieved a competitive performance on this dataset compared to other state-of-the-art models.

#According to the original paper introducing the TyDiQA dataset ("TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages"), the mT5-base model achieved an F1 score of 51.5 on the Telugu portion of the dataset, which is higher than the average F1 score of 48.3 for all models evaluated on the Telugu portion of the dataset.

#The mT5-base-finetuned-tydiQA-question-generation model is a variant of the mT5-base model that has been fine-tuned specifically for question generation on the TyDiQA dataset, which means that it may have achieved even better performance on this task. However, the performance of the mT5-base-finetuned-tydiQA-question-generation model specifically on the Telugu portion of the TyDiQA dataset is not reported in the original paper.

#Overall, the mT5-base-finetuned-tydiQA-question-generation model is a high-quality model that has been trained on a diverse range of languages, including Telugu, and it should perform well on generating questions in Telugu based on the provided context and answer.
  
tokenizer = AutoTokenizer.from_pretrained("Narrativa/mT5-base-finetuned-tydiQA-question-generation")
model = AutoModelForSeq2SeqLM.from_pretrained("Narrativa/mT5-base-finetuned-tydiQA-question-generation")

context = "హైదరాబాద్ నగరం భారత దేశంలో ఉన్నది. ఇది తెలంగాణ రాష్ట్రంలో ఉన్నది."
answer = "తెలంగాణ రాష్ట్రం"

input_text = f"కంటెక్స్ట్: {context} జవాబు: {answer}"

inputs = tokenizer(input_text, return_tensors="pt", padding=True)

output = model.generate(inputs["input_ids"], max_length=64, num_beams=4, early_stopping=True)

generated_question = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_question)



##finetuing the question answering model on the synthetic dataset

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import torch

# Define the path to the TyDiQA dataset in Telugu
dataset_path = "/content/train.csv"

# Load the pre-trained Telugu tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/content/train.csv")
model = AutoModelForQuestionAnswering.from_pretrained("/content/train.csv")

# Load the TyDiQA dataset in Telugu
dataset = torch.load(dataset_path)

# Split the dataset into training, validation, and testing sets
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Define the training arguments
training_args = TrainingArguments(
    output_dir="/content/model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the test set
eval_results = trainer.evaluate(test_dataset)

# Save the fine-tuned model
model.save_pretrained("/content/model")
tokenizer.save_pretrained("/content/model")