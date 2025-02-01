from groq import Groq


API_KEY ='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'
client = Groq(api_key=API_KEY)

chat_completion = client.chat.completions.create(
  messages=[
        {
            "role": "user",
            "content": "Why are we learning classical digital signal procesisng at school when data driven neural network based alternatives perform far better?",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)

#using a env manager that has groq installed and that apikey is my personal one that im gonna push to 
#github just for this one time i swear i never do this i swear