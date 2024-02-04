# main.py
import asyncio
import audioop
import base64
import io
import json
import os
from typing import Annotated
import openai
import uvicorn
import vosk
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydub import AudioSegment
from pyngrok import ngrok
from starlette.responses import Response
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect

import audioop_compat

app = FastAPI()
templates = Jinja2Templates(directory="templates")

if load_dotenv():
    print('Loading .env variables')

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
TWILIO_ACCOUNT_SID = os.environ['TWILIO_ACCOUNT_SID']
TWILIO_AUTH_TOKEN = os.environ['TWILIO_AUTH_TOKEN']
ELEVENLABS_API_KEY = os.environ['ELEVENLABS_API_KEY']

PORT = int(os.environ.get('PORT', 8000))
ELEVENLABS_VOICE_ID = os.environ.get('ELEVENLABS_VOICE_ID', 'pNInz6obpgDQGcFmaJgB')  # '21m00Tcm4TlvDq8ikWAM'
VOSK_MODEL_PATH = os.environ.get('VOSK_MODEL_PATH', './vosk-model-small-en-us-0.15')

openai.api_key = OPENAI_API_KEY
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
model = vosk.Model(VOSK_MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("call_page.html", {"request": request})


@app.post('/call', response_class=HTMLResponse)
async def handle_incoming_calls(request: Request, From: str = Form(...)):
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f'wss://{request.headers.get("host")}/stream')
    response.append(connect)
    response.say('Please leave a message.')
    print(f'Incoming call from {From}')
    return Response(content=str(response), media_type='text/xml')


async def wait_for_user_input(websocket):
    rec = vosk.KaldiRecognizer(model, 16000)
    while True:
        json_data = await websocket.receive_text()
        data = json.loads(json_data)
        if data['event'] == 'start':
            print('Streaming is starting')
        elif data['event'] == 'stop':
            print('\nStreaming has stopped')
            return
        elif data['event'] == 'media':
            stream_sid = data['streamSid']
            audio = base64.b64decode(data['media']['payload'])
            audio = audioop_compat.ulaw2lin(audio, 2)
            audio = audioop.ratecv(audio, 2, 1, 8000, 16000, None)[0]
            if rec.AcceptWaveform(audio):
                r = json.loads(rec.Result())
                print(f'Accepted speech: {r["text"]}')
                return r['text'], stream_sid


@app.websocket('/stream')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    messages = [{'role': 'system', 'content': 'You are on a phone call with the user.'}]
    while True:
        user_input, stream_sid = await wait_for_user_input(websocket)
        messages.append({'role': 'user', 'content': user_input, })
        await chat_completion(messages, websocket, stream_sid, model='gpt-3.5-turbo')


async def text_chunker(chunks):
    """Split text into chunks, ensuring to not break sentences."""
    splitters = ('.', ',', '?', '!', ';', ':', '—', '-', '(', ')', '[', ']', '}', ' ')
    buffer = ''

    async for text in chunks:
        if buffer.endswith(splitters):
            yield buffer + ' '
            buffer = text
        elif text.startswith(splitters):
            yield buffer + text[0] + ' '
            buffer = text[1:]
        else:
            buffer += text

    if buffer:
        yield buffer + ' '


async def stream(audio_stream, twilio_ws, stream_sid):
    async for chunk in audio_stream:
        if chunk:
            audio = AudioSegment.from_file(io.BytesIO(chunk), format='mp3')
            if audio.channels == 2:
                audio = audio.set_channels(1)
            resampled = audioop.ratecv(audio.raw_data, 2, 1, audio.frame_rate, 8000, None)[0]
            audio_segment = AudioSegment(data=resampled, sample_width=audio.sample_width, frame_rate=8000, channels=1)
            pcm_audio = audio_segment.export(format='wav')
            pcm_data = pcm_audio.read()
            ulaw_data = audioop.lin2ulaw(pcm_data, audio.sample_width)
            message = json.dumps({'event': 'media', 'streamSid': stream_sid,
                                  'media': {'payload': base64.b64encode(ulaw_data).decode('utf-8'), }})
            await twilio_ws.send_text(message)


async def text_to_speech_input_streaming(voice_id, text_iterator, twilio_ws, stream_sid):
    uri = f'wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_monolingual_v1&optimize_streaming_latency=3'

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({'text': ' ', 'voice_settings': {'stability': 0.5, 'similarity_boost': True},
                                         'xi_api_key': ELEVENLABS_API_KEY, }))

        async def listen():
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    if data.get('audio'):
                        yield base64.b64decode(data['audio'])
                    elif data.get('isFinal'):
                        break
                except websockets.exceptions.ConnectionClosed:
                    print('Connection closed')
                    break

        listen_task = asyncio.create_task(stream(listen(), twilio_ws, stream_sid))

        async for text in text_chunker(text_iterator):
            await websocket.send(json.dumps({'text': text, 'try_trigger_generation': True}))

        await websocket.send(json.dumps({'text': ''}))

        await listen_task


async def chat_completion(messages, twilio_ws, stream_sid, model='gpt-4'):
    response = await openai.ChatCompletion.acreate(model=model, messages=messages, temperature=1, stream=True,
                                                   max_tokens=50)

    async def text_iterator():
        full_resp = []
        async for chunk in response:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                content = delta['content']
                print(content, end=' ', flush=True)
                full_resp.append(content)
                yield content
            else:
                print('<end of gpt response>')
                break

        messages.append({'role': 'assistant', 'content': ' '.join(full_resp), })

    await text_to_speech_input_streaming(ELEVENLABS_VOICE_ID, text_iterator(), twilio_ws, stream_sid)


if __name__ == '__main__':
    ngrok.set_auth_token(os.environ['NGROK_AUTH_TOKEN'])
    public_url = ngrok.connect(str(PORT), bind_tls=True).public_url
    number = twilio_client.incoming_phone_numbers.list()[0]
    number.update(voice_url=public_url + '/call')
    print(f'Waiting for calls on {number.phone_number}')
    uvicorn.run(app, host='0.0.0.0', port=PORT)
                   