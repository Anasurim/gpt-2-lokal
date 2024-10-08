import requests

url = "http://127.0.0.1:4000/generate"  

payload = {
    "prompt": "Erstell ein E-mail",  # Corrected key to 'prompt'
    "max_length": 100  
}

try:
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("Antwort vom Server:")
        print(response.json())  
    else:
        print(f"Fehler: {response.status_code} - {response.text}")

except requests.exceptions.RequestException as e:
    print(f"Ein Fehler ist aufgetreten: {e}")
