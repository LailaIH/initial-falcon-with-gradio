import requests

# Bing Search API settings
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
BING_API_KEY = "5dda97573975414987887a971a4a4b92"  

# Hugging Face Falcon API settings
FALCON_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
FALCON_API_TOKEN = "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"  


def fetch_bing_search_results(query, count=3):
    """
    Fetches search results from Bing Search API.
    
    Args:
        query (str): The search query.
        count (int): Number of results to fetch.
    
    Returns:
        str: Combined snippet of search results.
    """
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": count}
    
    response = requests.get(BING_SEARCH_URL, headers=headers, params=params)
    
    if response.status_code == 200:
        results = response.json()
        #print("Bing API Response:", results) 
        
        # Check for 'webPages' and 'value' keys
        if isinstance(results, dict) and "webPages" in results and "value" in results["webPages"]:
            snippets = [item["snippet"] for item in results["webPages"]["value"]]
            return " ".join(snippets)
        else:
            raise Exception("Unexpected response structure or no results found.")
    else:
        raise Exception(f"Bing API failed: {response.status_code} - {response.text}")

def query_falcon_model(prompt, max_tokens=200, temperature=0.7):
    """
    Calls Hugging Face Falcon model API.
    
    Args:
        prompt (str): The text prompt for the model.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
    
    Returns:
        str: Generated response from the model.
    """
    headers = {"Authorization": f"Bearer {FALCON_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }
    }
    
    response = requests.post(FALCON_API_URL, headers=headers, json=payload)
    try:
        response_json = response.json()
        
        # Ensure the response is a list and contains the expected key
        if isinstance(response_json, list) and "generated_text" in response_json[0]:
            return response_json[0]["generated_text"]
        else:
            raise Exception(f"Unexpected response structure: {response_json}")
    except ValueError as e:
        raise Exception(f"Failed to parse JSON: {e}")
    except KeyError as e:
        raise Exception(f"Key error in response: {e}")



def enhanced_query_with_search(user_query):
    """
    Combines Bing Search results with user query and sends to Falcon model.
    
    Args:
        user_query (str): The user's query.
    
    Returns:
        str: Final answer to the user's question.
    """
    # Step 1: Get search results
    search_results = fetch_bing_search_results(user_query)
    
    # Step 2: Combine search results with user query
    enriched_prompt = f"Using the following context from the web:\n{search_results}\nAnswer this query:\n{user_query}\nAnswer is: "
    
    # Step 3: Query Falcon model
    falcon_response = query_falcon_model(enriched_prompt)
    
    # Step 4: Post-process the Falcon response to extract the answer
    # Assuming the response is a coherent sentence with the answer directly at the beginning.
    if isinstance(falcon_response, str):
        # Simplify the response to provide just the answer
        final_answer = falcon_response.strip()
        return final_answer
    else:
        raise Exception("Falcon model response is not a valid string.")



# Example Usage
if __name__ == "__main__":
    query = "can you help me to conduct a market study analysis?"
    try:
        result = enhanced_query_with_search(query)
        print(result)
    except Exception as e:
        print("Error:", e)
