import os, json
import google.generativeai as genai

from FeatureMetrics import calculate_feature_metrics
from helper import remove_header_text, remove_footer_text

def main():
    current_directory = os.getcwd()
    print(f"\nCurrent Working Directory: {current_directory}")
    
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    print(f"Parent Directory: {parent_directory}\n")
    
    # Parse config file to get API key
    with open('src/config.json', 'r') as file:
        config = json.load(file)
    
    # Setup API
    os.environ['GOOGLE_API_KEY'] = config.get('api_key')
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    model = genai.GenerativeModel("gemini-1.5-pro")
    
    # load writing sample 
    with open('datasets/ProjectGutenbergTop2000/books/3.txt', 'r', encoding='utf-8-sig') as file:
        text = file.read()
    
    # Remove header text
    text = remove_header_text(text)
    # Remove footer text
    text = remove_footer_text(text)
    
    # Extract feature metrics
    metrics = calculate_feature_metrics(text)
    print(metrics)
    
    # Prompt model to generate content with extracted metrics
    prompt = f"{metrics}\n Using these stylistic metrics write a paragraph about how you enjoy going on walks through the woods"
    response = model.generate_content(prompt)
    
    print(response)
    
if __name__ == "__main__":
    main()