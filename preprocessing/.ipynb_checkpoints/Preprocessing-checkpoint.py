import parsivar
import pandas as pd
import numpy as np
import parsivar
from tqdm import tqdm
import re
import string
from hazm import Stemmer


class Preprocessing:
    """
    Persian text cleaner with additional features on Parsivar package.
    """

    def __init__(self, data_folder):
        self.parsivar_normalizer = parsivar.Normalizer(statistical_space_correction=True)
        
        self.stopwords = []
        with open(data_folder + 'Persian_StopList.txt', encoding="utf8") as f:
            self.stopwords.append(f.readlines())
            
        self.stopwords = [word.replace('\n', '') for word in self.stopwords[0]]    
        self.cardinal = ['صفر','یک','دو','سه','چهار','پنج','شش','هفت','هشت','نه','ده']
        self.ordinal = ['صفرم','یکم','دوم','سوم','چهارم','پنجم','ششم','هفتم','هشتم','نهم','دهم']
    
        self.correctionConcepts = np.array(pd.read_csv(data_folder + 'CorrectionConcepts.csv').iloc[1:])
        
        

        self.char_mappings = {
            "А": "a",
            "В": "b",
            "C": "c",
            "D": "d",
            "Е": "e",
            "F": "f",
            "G": "g",
            "Н": "h",
            "I": "i",
            "J": "j",
            "K": "k",
            "L": "l",
            "M": "m",
            "N": "n",
            "O": "o",
            "P": "p",
            "Q": "q",
            "R": "r",
            "S": "s",
            "Т": "t",
            "U": "u",
            "V": "v",
            "W": "w",
            "X": "x",
            "Y": "y",
            "Z": "z",
            "к": "k",
            "м": "m",
            "о": "o",
            "р": "p",
            "ڈ": "د",
            "ڇ": "چ",
            # Persian numbers (will be replaced by english one)
            "۰": "0",
            "۱": "1",
            "۲": "2",
            "۳": "3",
            "۴": "4",
            "۵": "5",
            "۶": "6",
            "۷": "7",
            "۸": "8",
            "۹": "9",
            ".": ".",
            # Arabic numbers (will be replaced by english one)
            "٠": "0",
            "١": "1",
            "٢": "2",
            "٣": "3",
            "٤": "4",
            "٥": "5",
            "٦": "6",
            "٧": "7",
            "٨": "8",
            "٩": "9",
            # Special Arabic Characters (will be replaced by persian one)
            "ك": "ک",
            "ى": "ی",
            "ي": "ی",
            "ؤ": "و",
            "ئ": "ی",
            "إ": "ا",
            "أ": "ا",
            "آ": "ا",
            "ة": "ه",
            "ء": "ی",
            # French alphabet (will be replaced by english one)
            "à": "a",
            "ä": "a",
            "ç": "c",
            "é": "e",
            "è": "e",
            "ê": "e",
            "ë": "e",
            "î": "i",
            "ï": "i",
            "ô": "o",
            "ù": "u",
            "û": "u",
            "ü": "u",
            # Comma (will be replaced by dots for floating point numbers)
            ",": ".",
            # And (will be replaced by dots for floating point numbers)
            "&": " and ",
            # Vowels (will be removed)
            "ّ": "",  # tashdid
            "َ": "",  # a
            "ِ": "",  # e
            "ُ": "",  # o
            "ـ": "",  # tatvil
            # Spaces
            "‍": "",  # 0x9E -> ZERO WIDTH JOINER
            "‌": " ",  # 0x9D -> ZERO WIDTH NON-JOINER
            # Arabic Presentation Forms-A (will be replaced by persian one)
            "ﭐ": "ا",
            "ﭑ": "ا",
            "ﭖ": "پ",
            "ﭗ": "پ",
            "ﭘ": "پ",
            "ﭙ": "پ",
            "ﭞ": "ت",
            "ﭟ": "ت",
            "ﭠ": "ت",
            "ﭡ": "ت",
            "ﭺ": "چ",
            "ﭻ": "چ",
            "ﭼ": "چ",
            "ﭽ": "چ",
            "ﮊ": "ژ",
            "ﮋ": "ژ",
            "ﮎ": "ک",
            "ﮏ": "ک",
            "ﮐ": "ک",
            "ﮑ": "ک",
            "ﮒ": "گ",
            "ﮓ": "گ",
            "ﮔ": "گ",
            "ﮕ": "گ",
            "ﮤ": "ه",
            "ﮥ": "ه",
            "ﮦ": "ه",
            "ﮪ": "ه",
            "ﮫ": "ه",
            "ﮬ": "ه",
            "ﮭ": "ه",
            "ﮮ": "ی",
            "ﮯ": "ی",
            "ﮰ": "ی",
            "ﮱ": "ی",
            "ﯼ": "ی",
            "ﯽ": "ی",
            "ﯾ": "ی",
            "ﯿ": "ی",
            # Arabic Presentation Forms-B (will be removed)
            "ﹰ": "",
            "ﹱ": "",
            "ﹲ": "",
            "ﹳ": "",
            "ﹴ": "",
            "﹵": "",
            "ﹶ": "",
            "ﹷ": "",
            "ﹸ": "",
            "ﹹ": "",
            "ﹺ": "",
            "ﹻ": "",
            "ﹼ": "",
            "ﹽ": "",
            "ﹾ": "",
            "ﹿ": "",
            # Arabic Presentation Forms-B (will be replaced by persian one)
            "ﺀ": "ی",
            "ﺁ": "ا",
            "ﺂ": "ا",
            "ﺃ": "ا",
            "ﺄ": "ا",
            "ﺅ": "و",
            "ﺆ": "و",
            "ﺇ": "ا",
            "ﺈ": "ا",
            "ﺉ": "ی",
            "ﺊ": "ی",
            "ﺋ": "ی",
            "ﺌ": "ی",
            "ﺍ": "ا",
            "ﺎ": "ا",
            "ﺏ": "ب",
            "ﺐ": "ب",
            "ﺑ": "ب",
            "ﺒ": "ب",
            "ﺓ": "ه",
            "ﺔ": "ه",
            "ﺕ": "ت",
            "ﺖ": "ت",
            "ﺗ": "ت",
            "ﺘ": "ت",
            "ﺙ": "ث",
            "ﺚ": "ث",
            "ﺛ": "ث",
            "ﺜ": "ث",
            "ﺝ": "ج",
            "ﺞ": "ج",
            "ﺟ": "ج",
            "ﺠ": "ج",
            "ﺡ": "ح",
            "ﺢ": "ح",
            "ﺣ": "ح",
            "ﺤ": "ح",
            "ﺥ": "خ",
            "ﺦ": "خ",
            "ﺧ": "خ",
            "ﺨ": "خ",
            "ﺩ": "د",
            "ﺪ": "د",
            "ﺫ": "ذ",
            "ﺬ": "ذ",
            "ﺭ": "ر",
            "ﺮ": "ر",
            "ﺯ": "ز",
            "ﺰ": "ز",
            "ﺱ": "س",
            "ﺲ": "س",
            "ﺳ": "س",
            "ﺴ": "س",
            "ﺵ": "ش",
            "ﺶ": "ش",
            "ﺷ": "ش",
            "ﺸ": "ش",
            "ﺹ": "ص",
            "ﺺ": "ص",
            "ﺻ": "ص",
            "ﺼ": "ص",
            "ﺽ": "ض",
            "ﺾ": "ض",
            "ﺿ": "ض",
            "ﻀ": "ض",
            "ﻁ": "ط",
            "ﻂ": "ط",
            "ﻃ": "ط",
            "ﻄ": "ط",
            "ﻅ": "ظ",
            "ﻆ": "ظ",
            "ﻇ": "ظ",
            "ﻈ": "ظ",
            "ﻉ": "ع",
            "ﻊ": "ع",
            "ﻋ": "ع",
            "ﻌ": "ع",
            "ﻍ": "غ",
            "ﻎ": "غ",
            "ﻏ": "غ",
            "ﻐ": "غ",
            "ﻑ": "ف",
            "ﻒ": "ف",
            "ﻓ": "ف",
            "ﻔ": "ف",
            "ﻕ": "ق",
            "ﻖ": "ق",
            "ﻗ": "ق",
            "ﻘ": "ق",
            "ﻙ": "ک",
            "ﻚ": "ک",
            "ﻛ": "ک",
            "ﻜ": "ک",
            "ﻝ": "ل",
            "ﻞ": "ل",
            "ﻟ": "ل",
            "ﻠ": "ل",
            "ﻡ": "م",
            "ﻢ": "م",
            "ﻣ": "م",
            "ﻤ": "م",
            "ﻥ": "ن",
            "ﻦ": "ن",
            "ﻧ": "ن",
            "ﻨ": "ن",
            "ﻩ": "ه",
            "ﻪ": "ه",
            "ﻫ": "ه",
            "ﻬ": "ه",
            "ﻭ": "و",
            "ﻮ": "و",
            "ﻯ": "ی",
            "ﻰ": "ی",
            "ﻱ": "ی",
            "ﻲ": "ی",
            "ﻳ": "ی",
            "ﻴ": "ی",
            "ﻵ": "لا",
            "ﻶ": "لا",
            "ﻷ": "لا",
            "ﻸ": "لا",
            "ﻹ": "لا",
            "ﻺ": "لا",
            "ﻻ": "لا",
            "ﻼ": "لا",
        }

        self.valid_chars = [
            " ",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "آ",
            "ئ",
            "ا",
            "ب",
            "ت",
            "ث",
            "ج",
            "ح",
            "خ",
            "د",
            "ذ",
            "ر",
            "ز",
            "س",
            "ش",
            "ص",
            "ض",
            "ط",
            "ظ",
            "ع",
            "غ",
            "ف",
            "ق",
            "ل",
            "م",
            "ن",
            "ه",
            "و",
            "پ",
            "چ",
            "ژ",
            "ک",
            "گ",
            "ی",
            "؟",
            "،",
            ".",
            ",",
            ";",
            ":",
            "?",
            "!",
            "?",
            "-",
            "[",
            "]",
            "(",
            ")",
            "{",
            "}",
            "‘",
            "“",
        ]
    
    @staticmethod
    def _replace_rep(t):
        """Replace repetitions at the character level: ccc -> c"""

        def __replace_rep(m):
            c, cc = m.groups()
            return str(c)

        re_rep = re.compile(r"(\S)(\1{2,})")
        return re_rep.sub(__replace_rep, t)

    @staticmethod
    def _replace_wrap(t):
        """Replace word repetitions: word word word -> word"""

        def __replace_wrap(m):
            c, cc = m.groups()
            return str(c)

        re_wrap = re.compile(r"(\b\w+\W+)(\1{2,})")
        return re_wrap.sub(__replace_wrap, t)

    def format_punc(self, text):
        text = re.sub("""((?<=[A-Za-z\d()])\.(?=[A-Za-z]{2})|(?<=[A-Za-z]{2})\.(?=[A-Za-z\d]))""", '. ', text)
        text = re.sub("""((?<=[A-Za-z\d()]),(?=[A-Za-z]{2})|(?<=[A-Za-z]{2}),(?=[A-Za-z\d]))""", ', ', text)
        text = re.sub("""((?<=[A-Za-z\d{}])\.(?=[A-Za-z]{2})|(?<=[A-Za-z]{2})\.(?=[A-Za-z\d]))""", '. ', text)
        text = re.sub("""((?<=[A-Za-z\d{}]),(?=[A-Za-z]{2})|(?<=[A-Za-z]{2}),(?=[A-Za-z\d]))""", ', ', text)
        text = re.sub("""((?<=[A-Za-z\d[]])\.(?=[A-Za-z]{2})|(?<=[A-Za-z]{2})\.(?=[A-Za-z\d]))""", '. ', text)
        text = re.sub("""((?<=[A-Za-z\d[]]),(?=[A-Za-z]{2})|(?<=[A-Za-z]{2}),(?=[A-Za-z\d]))""", ', ', text)
        text = re.sub(r'(?<=[،؟;:?!])(?=\S)', r' ', text)  # add space after punctuations
        text = re.sub(r'\s([،؟.,;:?!"](?:\s|$))', r'\1', text)  # remove space before punctuations
        text = re.sub(r"\s?(\(.*?\))\s?", r" \1 ", text)  # Add space before and after ( and )
        text = re.sub(r"\s?(\{.*?\})\s?", r" \1 ", text)  # Add space before and after { and }
        text = re.sub(r"\s?(\[.*?])\s?", r" \1 ", text)  # Add space before and after [ and ]
        # Remove space after & before '(' and '[' and '{'
        text = re.sub(r'(\s([?,.!"]))|(?<=[\[(\{])(.*?)(?=[)\]\}])', lambda x: x.group().strip(), text)
        text = re.sub(r'[.,;:?!]+(?=[.,;:?!])', '', text)  # Replace multiple punctuations with last one
        # text = re.sub(r'(?<=-)\s*|\s*(?=-)', '', text)  # Remove space before and after hyphen
        # text = re.sub(r'(?<=/)\s*|\s*(?=/)', '', text)  # no space before or after the forward slash /
        # text = re.sub('([&@])', r' \1 ', text)  # Space before and after of "&" and "@"
        text = re.sub(' +', ' ', text)  # Remove multiple space
        return text

    def normalize_text(self, x):
        """normalize a sentence"""

        x = str(x)
        x = self.parsivar_normalizer.normalize(x)  # apply `parsivar` normalizations
        x = re.sub(r"[\u200c\r\n]", " ", x)  # remove half space and new line characters
        x = x.lower()
        x = "".join(
            [self.char_mappings[xx] if xx in self.char_mappings else xx for xx in x]
        )  # substitute bad characters with appropriate ones
        x = re.sub(
            r"[^{}]".format("".join(self.valid_chars)), " ", x
        )  # just keep valid characters and substitute others with space
        x = re.sub(r"[a-z]+", r" \g<0> ", x)  # put space around words and numbers
        x = re.sub(r"[0-9]+", r" \g<0> ", x)  # put space around words and numbers
        x = re.sub(r"\s+", " ", x)  # remove more than one white spaces with space
        x = self._replace_rep(x)
        x = self._replace_wrap(x)
        x = self.format_punc(x)
        
        for cc in self.correctionConcepts:
            x = x.replace(cc[0], cc[1])
            
        return x.strip()

    def normalize_texts(self, text, use_tqdm=False):
        """normalize list of sentences"""

        if use_tqdm:
            text = [self.normalize_text(x) for x in tqdm(text)]
        else:
            text = [self.normalize_text(x) for x in text]
        return text
    

    def normalize_numbers(self, split_report):

            if((split_report[1] in self.cardinal) and (split_report[2] not in self.ordinal)):
                return str(self.cardinal.index(split_report[1]))      
            
            elif((split_report[1] in self.ordinal) and (split_report[0] not in self.cardinal)):
                return str(self.ordinal.index(split_report[1]))
                            
            else:
                return split_report[1]
                
    
    
    
    def complete_normalization(self, text, use_tqdm=False):
        
            text = self.normalize_text(text)
            text = ' '.join([word for word in text.split() if not word in self.stopwords])
            text = text.translate(str.maketrans('', '', string.punctuation.replace(':','').replace('.','')))
            text = re.sub(' +', ' ', text).strip()
            text_tmp = 'tmp ' + text + ' tmp'
            split_report = text_tmp.split()
            text = []
            stemmer = Stemmer()

            for i in range(1,len(split_report)-1):
#                     split_report[i] = stemmer.stem(split_report[i])
                    text.append(self.normalize_numbers(split_report[i-1:i+2]))
            
            
            return ' '.join(text)
       
    
    
    def cleaner(self, sample):
        sample = sample.replace('نظریه ی','نظریه')
        sample = ' '.join(sample.split())
        sample = sample.replace(' :',':').replace(': ',':').split('رادیوگرافی')[1:]
        sample = [x + 'رادیوگرافی' for x in sample if x]
        sample = ' '.join(sample)
        sample = sample.replace('گزارش و نظریه رادیولوژیست:','').replace('گزارش و نظریه رادیولوژیست','').split('راديولوژيست:')[0].replace('/ج','').replace('/ن','').strip()
        sample = ' '.join(sample.split())   
        return sample
    
    
    def sentence_tokenizer(self, sample):
    
        my_tokenizer = parsivar.Tokenizer()
        my_tagger = parsivar.POSTagger(tagging_model="wapiti")  
        tags = my_tagger.parse(my_tokenizer.tokenize_words(sample.translate(str.maketrans('', '', string.punctuation))))
        my_chunker = parsivar.FindChunks()
        chunks = my_chunker.chunk_sentence(tags)
        
        
        all_chunks = my_chunker.convert_nestedtree2rawstring(chunks)
        all_chunks = all_chunks.split(']')
        func = lambda x: x.replace('VP','').replace(']','').replace('[','').strip() if ('VP' in x) else None
        verbs = np.array([func(tag) for tag in all_chunks])
        verbs = set(verbs[verbs != None])

        for verb in verbs:
            sample = sample.replace(verb+' ', verb + '.')
        

        sample = sample.replace(': ',' tag .')        
        sample = sample.replace('..','.')
        sample = sample.replace('. .','.')
        
        sentences = my_tokenizer.tokenize_sentences(sample.strip())
        
        return sentences