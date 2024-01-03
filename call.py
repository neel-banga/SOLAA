from twilio.rest import Client

# Your Twilio Account SID and Auth Token
account_sid = 'AC6af5c42f995469138ec3b011fec4c2ab'
auth_token = 'b820cdbf6d8e584a1ea794b71fe4e5e5'

# Create a Twilio client
client = Client(account_sid, auth_token)

# Twilio phone number and recipient's phone number
from_number = '+18332845676'
to_number = '+15104954014'

# URL to TwiML script that will handle the call
twiml_url = 'http://example.com/your-twiml-script'

# Make a voice call
call = client.calls.create(
    to=to_number,
    from_=from_number,
    url=twiml_url
)

print(f"Call SID: {call.sid}")
