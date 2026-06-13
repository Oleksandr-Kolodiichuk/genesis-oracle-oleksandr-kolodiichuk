import os
from google import genai
from PIL import Image

def main():
    # Instantiate the client (automatically uses GEMINI_API_KEY from environment)
    client = genai.Client()

    # Path to the generated plot
    image_path = "data/audit_target.png"

    # Attempt to load the image using PIL
    try:
        target_image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found. Run generate_signals.py first.")
        return

    # Define the Visual Detective prompt exactly as requested
    prompt = (
        "You are a Visual Detective. Analyze this dynamic wave signal plot. "
        "1. Find the visual anomaly (an ugly, high-frequency clipping/amplitude saturation artifact). "
        "2. Guess the exact X-axis region (time in seconds) where the malfunction happened. "
        "3. Write a short, funny poem mocking the engineering team that allowed this bug to pass into production."
    )

    print("Uploading target image and pinging the visual oracle (gemini-2.0-flash)...\n")
    
    # Pass both the loaded image and the text prompt to the multimodal model
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[target_image, prompt]
    )

    # Output the detective's findings
    print(f"Visual Detective Response:\n{'-'*40}\n{response.text}\n{'-'*40}")

if __name__ == "__main__":
    main()