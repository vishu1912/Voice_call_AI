# elevenlabs_utils.py
import os
import requests

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Replace with your voice ID

HEADERS = {
    "xi-api-key": ELEVEN_API_KEY,
    "Content-Type": "application/json"
}

ELEVEN_TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"


def text_to_speech(text: str, filename="response.mp3") -> str:
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(ELEVEN_TTS_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Audio saved as {filename}")
        return filename
    else:
        print(f"❌ Error from ElevenLabs: {response.status_code} - {response.text}")
        return ""


# Example usage (for testing only)
if __name__ == "__main__":
    text_to_speech("Welcome to Cactus Club Cafe. What would you like to order today?", "welcome.mp3")
