from openai import AsyncOpenAI

def oai_client():
    try:
        with open("openai-api-key") as file:
            return AsyncOpenAI(api_key=file.read().strip())
    except FileNotFoundError:
        print("No openai-api-key file found, using dummy key")
        return AsyncOpenAI(api_key="DUMMY KEY")
