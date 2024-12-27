from langdetect import detect_langs
from transformers import MarianMTModel, MarianTokenizer
import re

class TranslationManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def is_valid_text(self, text):
        """
        Check if text is valid for language detection:
        - At least 20 characters
        - At least 40% of the characters should be letters
        """
        if len(text.strip()) < 20:
            return False
            
        # Contar letras vs total de caracteres
        total_chars = len(text.strip())
        letter_chars = len(re.findall(r'[a-zA-Z\u00C0-\u00FF]', text))  # Incluye letras ASCII y acentuadas
        
        # Al menos 40% deben ser letras
        if letter_chars / total_chars < 0.4:
            return False
            
        return True
    
    def detect_language(self, text):
        """
        Detect the language of the input text with confidence score
        Returns tuple (lang_code, confidence) or (None, 0) if detection fails
        Example: ('es', 0.9) for Spanish with 90% confidence
        """
        if not self.is_valid_text(text):
            return None, 0
            
        try:
            langs = detect_langs(text)
            if langs:
                # Get the most probable language
                most_probable = langs[0]
                return most_probable.lang, most_probable.prob
            return None, 0
        except:
            return None, 0
    
    def load_model(self, source_lang):
        """
        Load model and tokenizer for a specific language if not already loaded
        """
        if source_lang not in self.models:
            model_name = f'Helsinki-NLP/opus-mt-{source_lang}-en'
            self.models[source_lang] = MarianMTModel.from_pretrained(model_name)
            self.tokenizers[source_lang] = MarianTokenizer.from_pretrained(model_name)
    
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
            print(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails
    
    def translate_from_english(self, text, target_lang):
        """
        Translate text from English to target language using MarianMT
        """
        try:
            model_name = f'Helsinki-NLP/opus-mt-en-{target_lang}'
            if target_lang not in self.models:
                self.models[target_lang] = MarianMTModel.from_pretrained(model_name)
                self.tokenizers[target_lang] = MarianTokenizer.from_pretrained(model_name)
            
            model = self.models[target_lang]
            tokenizer = self.tokenizers[target_lang]
            
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            translated = model.generate(**inputs)
            translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            return translated_text
        
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails

def process_input(text):
    """
    Process input text: detect language and translate to English if needed
    Returns: (translated_text, original_language)
    Only translates if language is detected with confidence >= 0.85
    """
    manager = TranslationManager()
    detected_lang, confidence = manager.detect_language(text)
    
    if not detected_lang or confidence < 0.85:
        return text, None
    
    if detected_lang == 'en':
        return text, 'en'
    
    translated = manager.translate_to_english(text, detected_lang)
    return translated, detected_lang

def process_output(text, target_lang):
    """
    Process output text: translate from English to target language if needed
    """
    if not target_lang or target_lang == 'en':
        return text
    
    manager = TranslationManager()
    translated = manager.translate_from_english(text, target_lang)
    return translated
