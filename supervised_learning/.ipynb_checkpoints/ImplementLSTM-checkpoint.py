import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from TrainLSTM import LSTMModel  # Import your custom LSTM model

### FOR YOU TO CHANGE: CHANGE THIS 
new_data = pd.read_csv('../data/your_new_never_before_seen_data.csv')

# Tokenize the 'thought_process' column using the same pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenized_data = tokenizer(new_data['thought_process'].tolist(), truncation=True, padding=True, return_tensors='pt', max_length=128)
input_ids = tokenized_data['input_ids']
attention_mask = tokenized_data['attention_mask']

# Convert the tokenized data to PyTorch tensors
new_dataset = TensorDataset(input_ids, attention_mask)

# Create a DataLoader for the new data
batch_size = 16  # Adjust batch size as needed
new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)


### FOR YOU TO CHANGE: CHANGE THIS TO MATCH WHATEVER PARAMS YOU SETTLE ON IN THE TRAINING CODE
# Initialize the LSTM model
input_dim = 128  
hidden_dim = 64
num_layers = 2
output_dim = 1
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

# Load the trained model weights
model.load_state_dict(torch.load('../weights/trained_model_weights.pth'))

# Put the model in evaluation mode
model.eval()

# Perform inference on the new data
predictions = []
with torch.no_grad():
    for input_ids_batch, attention_mask_batch in new_loader:
        outputs = model(input_ids_batch, attention_mask_batch)
        predictions.extend(outputs.tolist())

# Post-processing: Adjust predictions as needed to generate quality scores
quality_scores = [round(score[0], 2) for score in predictions]

# Add the quality scores to the new data DataFrame
new_data['quality'] = quality_scores

new_data.to_csv('../data/new_data_with_scores.csv', index=False)