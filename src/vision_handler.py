import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI() # Initializes OpenAI client with API key from environment variable

def describe_image(image_path: str) -> str:
    """
    Uses OpenAI's GPT-4 Vision model to generate a description for an image.
    """
    if not os.path.exists(image_path):
        return f"Image file not found at {image_path}"
    
    try:
        with open(image_path, "rb") as image_file:
            response = client.chat.completions.create(
                model="gpt-4o", # Or gpt-4-vision-preview
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image concisely, focusing on any charts, graphs, or key information. What is its main subject or purpose? Make it useful for search."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_file.read().hex()}", "detail": "low"}} # You might need to base64 encode this
                        ],
                    }
                ],
                max_tokens=150,
            )
        description = response.choices[0].message.content
        return description
    except Exception as e:
        print(f"Error describing image {image_path}: {e}")
        return f"Could not describe image due to an error: {str(e)}"

# A placeholder for local vision models if OpenAI isn't used
def describe_image_local(image_path: str) -> str:
    """
    Placeholder for a local image description model (e.g., BLIP, MiniGPT-4).
    You would integrate a model like this if not using OpenAI Vision.
    """
    print(f"Using placeholder for local image description for {image_path}")
    return "A placeholder description of an image."
