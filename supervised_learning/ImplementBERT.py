import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Function to perform inference and save results to a CSV file
def infer_and_save_results(model, tokenizer, new_text_list, output_csv_filename):
    # Tokenize and encode the new text data
    new_encodings = tokenizer(new_text_list, truncation=True, padding=True, return_tensors='pt', max_length=512)

    # Use the loaded model for prediction
    model.eval()
    with torch.no_grad():
        inputs = {
            'input_ids': new_encodings['input_ids'],
            'attention_mask': new_encodings['attention_mask']
        }
        outputs = model(**inputs)
        predictions = outputs.logits.squeeze(1).tolist()

    # Create a DataFrame with results
    result_df = pd.DataFrame({'Text': new_text_list, 'Predicted Quality Score': predictions})

    # Print the results
    print(result_df)

    # Save the results to a CSV file
    result_df.to_csv(output_csv_filename, index=False)

# Load the fine-tuned model and tokenizer
save_directory = 'SIT_fine_tuned_model_directory'
model = BertForSequenceClassification.from_pretrained(save_directory)
tokenizer = BertTokenizer.from_pretrained(save_directory)

# New text data loaded from a CSV file
new_text_csv_filename = 'new_text_data.csv'
new_text_data = pd.read_csv(new_text_csv_filename)['Text'].tolist()

# Define the output CSV filename
output_csv_filename = 'results.csv'

# Perform inference and save results
infer_and_save_results(model, tokenizer, new_text_data, output_csv_filename)
