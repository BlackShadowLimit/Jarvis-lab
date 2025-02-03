import tts 
import stt
import brain
import warnings
warnings.filterwarnings("ignore", message="The current process just got forked, after parallelism has already been used")


jarvis = brain.ConversationJarvis()
great = "Welcome home Mr. Black"
print(great)
tts.speak(great)
while True:
    try:
        input_text = stt.main()
        if input_text:
            response = jarvis.generate_response(input_text)
            tts.speak(response)
            print(f"Jarvis: {response}")
    except KeyboardInterrupt:
        bye = "Goodbye Mr. Black, I always be here"
        print(bye)
        tts.speak(bye)
        break
