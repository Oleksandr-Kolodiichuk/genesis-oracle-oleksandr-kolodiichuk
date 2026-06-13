import os
from google import genai

def main():
    # Instantiate the client. It automatically retrieves GEMINI_API_KEY from the environment.
    client = genai.Client()

    # Define the required prompt comparing NumPy and JAX state management
    prompt = (
        "Explain the difference between a stateful NumPy random generation process "
        "and a stateless JAX PRNG split operation in exactly one highly sarcastic sentence."
    )

    print("Pinging the cognitive oracle (gemini-1.5-flash)...\n")
    
    # Ping the model
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )

    # Output the exact text response
    print(f"Model Response:\n{response.text}")

if __name__ == "__main__":
    main()