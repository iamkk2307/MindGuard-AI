"""
Voice Input Test Script
Tests if microphone and speech recognition are working properly
"""
import speech_recognition as sr

def test_microphone():
    print("=" * 60)
    print("MindGuard AI - Voice Input Diagnostics")
    print("=" * 60)
    
    # Test 1: Check microphone availability
    print("\n[Test 1] Checking microphone devices...")
    try:
        mic_list = sr.Microphone.list_microphone_names()
        print(f"✅ Found {len(mic_list)} microphone(s):")
        for i, mic in enumerate(mic_list):
            print(f"   {i}: {mic}")
    except Exception as e:
        print(f"❌ Error listing microphones: {e}")
        return
    
    # Test 2: Try to initialize microphone
    print("\n[Test 2] Initializing default microphone...")
    try:
        with sr.Microphone() as source:
            print("✅ Microphone initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing microphone: {e}")
        print("\nPossible fixes:")
        print("1. Make sure a microphone is connected")
        print("2. Check microphone permissions in Windows settings")
        print("3. Try reinstalling PyAudio with: pip install --upgrade pyaudio")
        return
    
    # Test 3: Record and recognize speech
    print("\n[Test 3] Recording speech...")
    print("🎤 Please speak now (you have 5 seconds)...")
    
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("✅ Audio recorded successfully")
            
        print("\n[Test 4] Converting speech to text...")
        text = recognizer.recognize_google(audio)
        print(f"✅ Recognition successful!")
        print(f"You said: '{text}'")
        
    except sr.WaitTimeoutError:
        print("❌ Timeout - no speech detected")
        print("Make sure you speak within 5 seconds after the prompt")
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        print("Try speaking more clearly or closer to the microphone")
    except sr.RequestError as e:
        print(f"❌ Could not connect to speech recognition service: {e}")
        print("Check your internet connection")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_microphone()
