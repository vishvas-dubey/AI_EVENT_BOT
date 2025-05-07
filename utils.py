import json

def check_registration(name):
    try:
        with open("participants.json", "r") as file:
            participants = json.load(file)
        return name.lower() in [p.lower() for p in participants.get("names", [])]
    except:
        return False