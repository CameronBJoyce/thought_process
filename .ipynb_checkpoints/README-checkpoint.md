# Thought Process Mapping

This project aims to map and understand the thought processes of individuals as they transition from one state to another. It involves using a Language Model to generate coherent narratives that describe the cognitive journey of a person in various scenarios and then building a supervised learning model to automatically grade written thought processes based on human labelled data.

### The process
**1. Our goal is to generate a list of possible thought processes. We will do this using text-da-vinci-003, a slightly older OpenAI model that is good for these instruct task.**

**2. In process_generation/ThoughtProcessGeneration.py we generate possible thought processes for different scenarios.**  
    - this code will have a static "goal" that is the same for every prompt we pass to OPEN AI  \
    - I generated sample data you will definitely want to change (and have be in CSV format). \
    - The columns are setting, scenario, context, test subject information, start state, end state. \
        - Check out the sample data for what I mean by that but feel free to add or subtract columns as you see fit, the code will handle whatever.
    - We set num_variations to be 3 (feel free to add or subtract number of variations for each situation).
    
**3. Now, some human researchers will need to score each thought process and save them in a csv format of 2 columns:** \
    1. thought process (the text used for a thought process) \
    2. quality score (0-10) score.
    
**4. This CSV forms our training dataset for the models. I have two here for you:** \
    1. LSTM -  neural network architecture designed to process and remember information over long sequences of data. \
    2. BERT -  deep learning framework that excels at handling sequential data by leveraging self-attention mechanisms. It has revolutionized natural language processing tasks by allowing models to efficiently capture contextual relationships among words in a sequence, making it highly effective for tasks like language translation and understanding.
    
**5. You have to use the csv the human researchers made to train these supervised models. You will have to do hyperparameter tuning. This code is here:** \
    - supervised_learning/TrainLSTM.py \
    - supervised_learning/TrainBERT.py
    
**6. Once you have that, you can experiment on new data with:** \
    - supervised_learning/ImplementLSTM.py </br>
    - supervised_learning/ImplementLSTM.py


### General Notes for BERT model:
- In the TrainBert.py the logits are the model's predicted quality scores
- The code defines a custom regression head that modifies the BERT model for regression by using a single output neuron.
- I used Mean Squared Error (MSE) Loss here to train the model. It measures the squared difference between predicted and actual values.
- I added other various evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Median Absolute Error (MedAE), and R-squared (R2) Score to assess the model's performance.
- Try your best to ensure quality scores in your dataset are reliable as the model will rely solely on the human quality scores.
- Depending on the dataset size and potential overfitting concerns, you may want to explore regularization techniques such as dropout or weight decay.
