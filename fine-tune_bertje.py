from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, AdamW
import torch
import torch.utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


'''Function for preprocessing data'''
def preprocess_function(examples):
    return tokenizer(
        examples["text"], 
        add_special_tokens=True,
        truncation=True
        )

'''Function for computing evaluation metrics'''
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1_result = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1_result
}

# Set seed
torch.manual_seed(42)

# GPU selction
if torch.cuda.is_available():    
   
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load dataset
dataset = load_dataset("GroNLP/dutch-cola")
# Modify dataset
dataset_mod = dataset.filter(lambda x: x["Original annotation"] in [None, "*"])
# Rename columns
dataset_mod = dataset_mod.rename_column("Sentence", "text")
dataset_mod = dataset_mod.rename_column("Acceptability", "label")
# Remove unecessary columns
dataset_mod = dataset_mod.remove_columns(["Source", "Original ID", "Original annotation", "Material added"])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")

# Tokenize dataset
tokenized_dataset = dataset_mod.map(preprocess_function, batched=True)
# Split dataset
tokenized_train = tokenized_dataset["train"]
tokenized_val = tokenized_dataset["validation"]
tokenized_test = tokenized_dataset["test"]

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model, run on selected device
model = AutoModelForSequenceClassification.from_pretrained("GroNLP/bert-base-dutch-cased", num_labels=2)
model.to(device)

optimizer = AdamW(
    model.parameters(), 
    lr=4e-5, 
    weight_decay=0.01, 
    betas=(0.9, 0.999), 
    eps=1e-8
)

# Arguments
training_args = TrainingArguments(
    output_dir='./results',
    logging_dir='logs',
    learning_rate=4e-5,
    weight_decay=0.01,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    logging_strategy='epoch',
    gradient_accumulation_steps=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)
)

# Train model
trainer.train()

# Evaluate model performance
trainer.evaluate(tokenized_test)

# Save the model
model.save_pretrained('./fine-tuned-bertje')