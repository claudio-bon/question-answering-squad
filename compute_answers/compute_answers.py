# -*- coding: utf-8 -*-

import argparse
import json
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, default_data_collator

from file_loader import generate_dataset
from preprocess import preprocess_eval
from postprocess import postprocess_eval



def main():
    ### Handling arguments ###
    parser = argparse.ArgumentParser(description="Given a json file formatted as the "+ \
                                                 "training set, creates a prediction file "+ \
                                                 "in the desired format")
    parser.add_argument('file_path', type=str,
                       help='Path to the .json file that holds the answer and related contexts')
    shell_args = parser.parse_args()
    ###########################
    
    
    ### Loading the dataset ###
    ds_original = load_dataset('json', data_files=shell_args.file_path, field='data')
    df = pd.DataFrame([value[1] for value in generate_dataset(ds_original, test=True)])
    
    dataset = Dataset.from_pandas(df)
    ###########################
    
    
    ### Prediction ###
    
    # Define HuggingFace architecture needed for inferece
    model_checkpoint = "distilbert-base-uncased"
    # Load the model from local files.
    model = AutoModelForQuestionAnswering.from_pretrained("squad_trained")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = default_data_collator
    trainer = Trainer(
            model,
            data_collator=data_collator,
            tokenizer=tokenizer,
    )
    
    max_length = 384 # Max length of the sequence
    stride = 128 # Overlap of the context
    
    #Preprocessing
    eval_features = dataset.map(
        preprocess_eval(tokenizer, max_length, stride),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    #Predict
    raw_predictions = trainer.predict(eval_features)
    #When predicting the Trainer class hides the features not used for the prediction
    #This line of codes brings those features back
    eval_features.set_format(type=eval_features.format["type"], columns=list(eval_features.features.keys()))
    
    #Postprocessing
    eval_predictions = postprocess_eval(
        dataset,
        eval_features,
        raw_predictions.predictions
    )
    ##################


    ### Saving the predictions ###
    with open('predictions.json', 'w') as fp:
        json.dump(eval_predictions, fp)
    ##############################
    
   
if __name__ == "__main__":
    main()