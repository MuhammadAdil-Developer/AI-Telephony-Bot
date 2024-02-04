Voice Chatbot with Twilio, OpenAI, and Eleven Labs
This project showcases a voice chatbot using Twilio, OpenAI's GPT-3.5, and Eleven Labs' Text-to-Speech API. The chatbot transcribes user speech during phone calls, generates responses using GPT-3.5, and streams the replies back in real-time.

Setup
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Create a .env file with credentials:

env
Copy code
OPENAI_API_KEY=your_openai_api_key
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
ELEVENLABS_API_KEY=your_elevenlabs_api_key
NGROK_AUTH_TOKEN=your_ngrok_auth_token
PORT=8000
ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id
VOSK_MODEL_PATH=./vosk-model-small-en-us-0.15
Download the Vosk Model from here and place it in the project directory.

Usage
Expose the local server:

bash
Copy code
ngrok authtoken your_ngrok_auth_token
ngrok http 8000
Update Twilio Phone Number:

Replace desired_phone_number_sid in the code with your Twilio phone number SID.

Run the application:

bash
Copy code
python your_app_file.py
Configure Twilio:

Update your Twilio incoming phone number's voice URL to the Ngrok public URL with /call endpoint.

Test the application:

Make a call to your Twilio phone number, and the chatbot will interact with you, transcribing your speech and generating real-time responses.

