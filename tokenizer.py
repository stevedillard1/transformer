import numpy as np
from jaxtyping import Int, Float
from collections import Counter
import unicodedata


class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self,
        text: str,
        process: bool = False,
    ) -> list[str]:
        if process:
            text = self.process_text(text)
        return text.split(' ')

    def encode(self,
            text: str|list[str],
    ) -> Int[np.ndarray, "n_tokens"]:
            if isinstance(text, str):
                text = self.tokenize(text)
            return np.array([self.vocab_dict[word] for word in text])

    def decode(self,
        encoded_text: list[int],
    ) -> str:
        return ' '.join(self.vocab_arr[i] for i in encoded_text)

    def process_set_tokenize_text(self, text: str):
        processed_text = self.process_text(text)
        tokenized_text = self.tokenize(processed_text)
        VOCAB_FREQ: Counter[str] = self.analyze_vocab(tokenized_text)
        self.vocab_arr: list[str] = [word for word, _ in VOCAB_FREQ.most_common()]
        self.vocab_dict: dict[str, int] = {word: i for i, word in enumerate(self.vocab_arr)}
        return tokenized_text
        
    def process_text(self,
            text: str,
            allowed_punctuation: str = "-.,;:!?()\"" + "".join(str(x) for x in range(10)),
            punctuation_convert: dict[str,str] = {'—': '-'},
        ) -> str:
            
            # replace some special characters which unicode won't normalize properly
            for char, replacement in punctuation_convert.items():
                text = text.replace(char, replacement)

            # if a line has ".jpg" in it, remove that line (this is specific to Don Quixote)
            text = '\n'.join(
                line 
                for line in text.split('\n')
                if '.jpg' not in line
            )

            # Normalize the string to decompose Unicode characters
            text = unicodedata.normalize('NFKD', text)

            # Encode to ASCII bytes, then decode back to string, ignoring errors
            text = text.encode('ascii', 'ignore').decode('ascii')

            # remove newlines and tabs
            text = text.replace('\n', ' ').replace('\t', ' ')


            # put spaces around allowed punctuation
            for char in allowed_punctuation:
                text = text.replace(char, f' {char} ')


            # remove leading and trailing spaces
            text = text.strip()

            # remove multiple spaces
            while '  ' in text:
                text = text.replace('  ', ' ')


            # remove all characters except (alphanumeric, allowed_punctuation, ' ')
            text = ''.join(
                (
                    char 
                    if (
                        char.isalnum() 
                        or char in allowed_punctuation 
                        or char == ' '
                    )
                    else ' '
                )
                for char in text 
            )

            # convert to lowercase
            text = text.lower()

            text = text.strip()

            return text

    def analyze_vocab(self,
        tokenized_text: list[str],
    ) -> Counter[str]:
        vocab_freq: Counter[str] = Counter(tokenized_text)
        return vocab_freq

            