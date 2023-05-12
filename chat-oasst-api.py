import requests
import json
import colorama

SERVER_IP = "10.0.0.18"
URL = f"http://{SERVER_IP}:5000/generate"

USERTOKEN = "<|prompter|>"
ENDTOKEN = "<|endoftext|>"
ASSISTANTTOKEN = "<|assistant|>"

def prompt(inp):
    data = {"text": inp}
    headers = {'Content-type': 'application/json'}

    response = requests.post(URL, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        return response.json()["generated_text"]
    else:
        return "Error:", response.status_code
    
history = ""
while True:
    inp = input(">>> ")
    context = history + USERTOKEN + inp + ENDTOKEN + ASSISTANTTOKEN
    output = prompt(context)
    history = output
    just_latest_asst_output = output.split(ASSISTANTTOKEN)[-1].split(ENDTOKEN)[0]
    # color just_latest_asst_output green in print:
    print(colorama.Fore.GREEN + just_latest_asst_output + colorama.Style.RESET_ALL)


   