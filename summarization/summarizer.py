from transformers import BartForConditionalGeneration, BartTokenizer
import torch

class TextSummarizer:
    def __init__(self):
        print("Cargando modelo de resumen BART...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(self.device)
        print(f"BART cargado en {self.device}")

    def summarize(self, text, max_length=150, min_length=50):
        """
        Summarize the given text using BART model
        """
        inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(self.device)
        
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Asegurar que el resumen termine en punto
        if not summary.endswith('.'):
            # Encontrar el último punto
            last_period = summary.rfind('.')
            if last_period != -1:
                # Si hay un punto, cortar hasta ahí
                summary = summary[:last_period + 1]
            else:
                # Si no hay punto, añadir uno
                summary = summary.rstrip() + '.'
        
        return summary

# Ejemplo de uso
if __name__ == "__main__":
    summarizer = TextSummarizer()
    
    # Texto de ejemplo
    text = """
    We present a Spitzer based census of the IC 348 nebula and embedded star cluster. Our Spitzer census supplemented by ground based spectra has added 42 class II T-Tauri sources to the cluster membership and identified ~20 class 0/I protostars. The population of IC 348 likely exceeds 400 sources after accounting statistically for unidentified diskless members. Our Spitzer census of IC 348 reveals a population of protostars that is anti-correlated spatially with the T-Tauri members, which comprise the centrally condensed cluster around a B star. The protostars are instead found mostly at the cluster periphery about 1 pc from the B star and spread out along a filamentary ridge. We find that the star formation rate in this protostellar ridge is consistent with that rate which built the exposed cluster while the presence of fifteen cold, starless, millimeter cores intermingled with this protostellar population indicates that the IC 348 nebula has yet to finish forming stars. We show that the IC 348 cluster is of order 3-5 crossing times old, and, as evidenced by its smooth radial profile and confirmed mass segregation, is likely relaxed. While it seems apparent that the current cluster configuration is the result of dynamical evolution and its primordial structure has been erased, our findings support a model where embedded clusters are built up from numerous smaller sub-clusters. Finally, the results of our Spitzer census indicate that the supposition that star formation must progress rapidly in a dark cloud should not preclude these observations that show it can be relatively long lived.
    """
    
    summary = summarizer.summarize(text)
    print("\nResumen generado:")
    print(summary)