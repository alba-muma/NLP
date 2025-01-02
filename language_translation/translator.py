from langdetect import detect_langs
from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def detect_language(self, text):
        """
        Detect the language of the input text
        Returns the language code (e.g., 'es' for Spanish, 'fr' for French)
        """
        try:
            deteccion = detect_langs(text)[0]
            return deteccion.lang, deteccion.prob
        except:
            return None, 0
    
    def load_model(self, source_lang):
        """
        Load model and tokenizer for a specific language if not already loaded
        """
        if source_lang not in self.models:
            model_name = f'Helsinki-NLP/opus-mt-{source_lang}-en'
            print(f"\nLoading model for {source_lang}...")
            self.models[source_lang] = MarianMTModel.from_pretrained(model_name)
            self.tokenizers[source_lang] = MarianTokenizer.from_pretrained(model_name)
            print("Model loaded successfully!")
    
    def translate_to_english(self, text, source_lang):
        """
        Translate text from source language to English using MarianMT
        """
        try:
            # Load model if not already loaded
            self.load_model(source_lang)
            
            # Get the model and tokenizer
            model = self.models[source_lang]
            tokenizer = self.tokenizers[source_lang]
            
            # Tokenize and translate
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            translated = model.generate(**inputs)
            
            # Decode the translation
            translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            return translated_text
        
        except Exception as e:
            return f"Translation error: {str(e)}"

def main():
    # Create translator instance
    translator = Translator()
    
    while True:
        # Get input text from user
        print("\nEnter the text you want to translate to English (or 'q' to quit):")
        text = input()
        
        if text.lower() == 'q':
            break
        
        # Detect language
        detected_lang, confidence = translator.detect_language(text)
        if detected_lang:
            print(f"\nDetected language: {detected_lang} con confianza de {confidence}")
            
            # If text is already in English, no need to translate
            if detected_lang == 'en':
                print("Text is already in English!")
                continue
            
            # Translate to English
            print("\nTranslating...")
            translation = translator.translate_to_english(text, detected_lang)
            print(f"\nTranslation to English: {translation}")
        else:
            print("Could not detect the language of the input text.")

if __name__ == "__main__":
    main()
