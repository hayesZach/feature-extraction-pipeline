import os
import google.generativeai as genai

from FeatureMetrics import calculate_feature_metrics
from FeatureExtraction import RemoveHeaderText, RemoveFooterText

def main():
    current_directory = os.getcwd()
    print(f"\nCurrent Working Directory: {current_directory}")
    
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    print(f"Parent Directory: {parent_directory}\n")
    
    # load writing sample
    with open('datasets/ProjectGutenbergTop2000/books/3.txt', 'r', encoding='utf-8-sig') as file:
        text = file.read()
    
    # Remove header text
    text = RemoveHeaderText(text)
    # Remove footer text
    text = RemoveFooterText(text)
    
    metrics = calculate_feature_metrics(text)
    print(metrics)
    
    # Setup API
    os.environ['GOOGLE_API_KEY'] = "API_KEY_HERE"
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    model = genai.GenerativeModel("gemini-1.5-pro")
    
    prompt = f"{metrics}\n Using these stylistic metrics write a paragraph about how you enjoy going on walks through the woods"
    response = model.generate_content(prompt)
    
    print(response)
    
if __name__ == "__main__":
    main()