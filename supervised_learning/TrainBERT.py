"""
So this code trains an BERT model to evaluate whether a thought process is "good" or not. So it takes in a though process and then a human-assigned quality score and builds a model to rate like the human did.

See ImplementBERT.py for implementation steps. 

The data is 2 columns (thought process and quality)
1. thought_process (a string representing an expanded thought process)
2. quality (how accurate is this).

So: 
thought_process, quality
Row 1:
I have an important deadline for a project tomorrow, but I find myself procrastinating. It's a classic case of present bias, where I prioritize short-term pleasure (e.g., watching TV) over long-term goals (completing the project). I need to exercise self-control and engage in time management strategies to overcome this challenge. , 7.8

Row 2: 
I'm considering a major career change, but I catch myself selectively seeking information that confirms my preexisting beliefs about the new career path. This is a clear case of confirmation bias. I need to actively seek out and consider contradictory information to make an unbiased and informed decision. ,8.2

...

Row N:
I'm at a team meeting, and the majority of my colleagues express a certain opinion about a project. I feel pressure to conform to the group's views, even though I have reservations. This is a classic case of normative social influence. I need to consider whether my concerns are valid and whether I should express them to encourage constructive discussion. ,7.5

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error

# FOR YOU TO CHANGE
data = pd.read_csv('../data/SIT_thought_process_dataset.csv')

# Split the data into training and testing sets with stratification
X = data['thought_process']
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

# Load a pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Tokenize and encode the text prompts
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt', max_length=512)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt', max_length=512)

# Create DataLoader for training and testing
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train.values, dtype=torch.float32))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(y_test.values, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialize optimizer and loss function (MSE for regression)
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.MSELoss()

# Training loop
epochs = 3  # Adjust as needed
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.squeeze(1)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, _ = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions.extend(outputs.logits.squeeze(1).tolist())

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)
medae = median_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Median Absolute Error (MedAE): {medae}")
print(f"R-squared (R2) Score: {r2}")

# Save the fine-tuned model and tokenizer to a directory
save_directory = 'SIT_fine_tuned_model_directory'
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)