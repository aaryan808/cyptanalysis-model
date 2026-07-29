import re
import math

class QuadgramScorer:
    # A simplified dictionary of top ~200 English quadgrams and their log probabilities
    # Values represent log10(probability).
    DEFAULT_QUADGRAMS = {
        'TION': -1.2, 'THAT': -1.5, 'WITH': -1.8, 'THER': -1.3, 'MENT': -2.0,
        'HERE': -2.1, 'TING': -2.2, 'IGHT': -2.3, 'FROM': -2.4, 'THIS': -1.9,
        'WHIC': -2.5, 'HICH': -2.5, 'THES': -2.4, 'HESE': -2.4, 'THEY': -2.1,
        'HAVE': -2.2, 'OULD': -2.3, 'WERE': -2.6, 'SOME': -2.5, 'BEEN': -2.6,
        'WHER': -2.7, 'TIME': -2.8, 'WHEN': -2.6, 'WHAT': -2.5, 'YOUR': -2.5,
        'MORE': -2.7, 'WILL': -2.6, 'THEI': -2.4, 'HEIR': -2.4, 'OTHE': -2.2,
        'INTH': -1.6, 'OFTHE': -1.1, 'ANDT': -1.7, 'NDTH': -1.7, 'DTHE': -1.7,
        'TTHE': -2.0, 'FTHE': -1.9, 'ONTH': -2.1, 'NTHE': -2.1, 'ATTH': -2.2,
        'TTHE': -2.2, 'STHE': -2.3, 'RTHE': -2.3, 'ETHE': -2.3, 'TOTHE': -1.8,
        'INGT': -2.4, 'NGTH': -2.4, 'ENTH': -2.4, 'EDTH': -2.5, 'CTIO': -2.6,
        'ATIO': -1.5, 'PRO': -2.8, 'CON': -2.6, 'COM': -2.6, 'PER': -2.7,
        'ALL': -2.7, 'FOR': -2.3, 'OUT': -2.6, 'HIS': -2.6, 'WAS': -2.6,
        'TER': -2.5, 'VER': -2.5, 'MAN': -2.8, 'THIN': -2.7, 'HING': -2.7,
        'OUGH': -2.4, 'SHO': -2.8, 'HOUL': -2.8, 'OULD': -2.4, 'EVER': -2.6,
        'VERY': -2.7, 'THAN': -2.6, 'MUST': -2.9, 'MUCH': -2.9, 'SUCH': -2.9,
        'OVER': -2.8, 'EVEN': -2.8, 'INTO': -2.8, 'ONLY': -2.8, 'ALSO': -2.8,
        'LIKE': -2.9, 'JUST': -2.9, 'THOU': -2.7, 'HOUS': -2.8, 'BECA': -2.9,
        'ECAU': -2.9, 'CAUS': -2.9, 'AUSE': -2.9, 'THEM': -2.6, 'THEN': -2.6,
        'WELL': -2.9, 'MAKE': -2.9, 'MADE': -2.9, 'CAME': -2.9, 'SAID': -2.9,
        'KNOW': -2.9, 'DOWN': -2.9, 'UPON': -2.9, 'WORK': -3.0, 'YEAR': -3.0,
        'GOOD': -3.0, 'THRO': -2.9, 'HROU': -2.9, 'ROUG': -2.9, 'UGH': -3.0,
        'PEOP': -3.0, 'EOPL': -3.0, 'OPLE': -3.0, 'GOVE': -3.0, 'OVER': -2.7,
        'VERN': -3.0, 'ERNM': -3.0, 'RNME': -3.0, 'NMEN': -3.0, 'MENT': -2.2,
        'ANTE': -3.0, 'PART': -3.0, 'STAT': -3.0, 'TATE': -3.0, 'WITH': -1.9,
        'AGAI': -3.0, 'GAIN': -3.0, 'AINS': -3.0, 'INST': -3.0, 'ANOT': -3.0,
        'NOTH': -3.0, 'HER': -2.4, 'AMER': -3.0, 'MERIC': -3.0, 'RICA': -3.0,
        'ICAN': -3.0, 'COUN': -3.0, 'OUNT': -3.0, 'UNTR': -3.0, 'NTRY': -3.0,
        'NATI': -3.0, 'TIONAL': -2.8, 'TION': -1.2, 'ONAL': -3.0, 'INTE': -3.0,
        'NTER': -2.9, 'ESTI': -3.0, 'STIN': -3.0, 'TING': -2.4, 'PRES': -3.0,
        'RESI': -3.0, 'ESID': -3.0, 'SIDE': -3.0, 'IDEN': -3.0, 'DENT': -3.0,
        'PORT': -3.0, 'ORTE': -3.0, 'RTED': -3.0, 'TION': -1.2, 'IONS': -2.5,
        'COMM': -3.0, 'MMUN': -3.0, 'MUNI': -3.0, 'UNIC': -3.0, 'NICA': -3.0,
        'ICAT': -3.0, 'CATI': -3.0, 'ATIO': -1.8, 'PUBL': -3.0, 'UBLI': -3.0,
        'BLIC': -3.0, 'GENE': -3.0, 'ENER': -3.0, 'NERA': -3.0, 'ERAL': -3.0
    }

    def __init__(self, filepath=None):
        """Load quadgram frequencies. If no file, generate from built-in data."""
        self.quadgrams = self.DEFAULT_QUADGRAMS.copy()
        self.floor = -10.0
        
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) == 2:
                            self.quadgrams[parts[0]] = float(parts[1])
            except Exception as e:
                print(f"Warning: Could not load {filepath}, using defaults. Error: {e}")

    def clean_text(self, text: str) -> str:
        """Clean input text to uppercase A-Z only."""
        return re.sub(r'[^A-Z]', '', text.upper())

    def score(self, text: str) -> float:
        """Return log-probability score of text based on quadgram frequencies."""
        cleaned = self.clean_text(text)
        if len(cleaned) < 4:
            return self.floor
        
        total_score = 0.0
        for i in range(len(cleaned) - 3):
            q = cleaned[i:i+4]
            total_score += self.quadgrams.get(q, self.floor)
            
        return total_score

    def score_per_char(self, text: str) -> float:
        """Return normalized score (per character)."""
        cleaned = self.clean_text(text)
        if len(cleaned) < 4:
            return self.floor
        return self.score(text) / (len(cleaned) - 3)


class EnglishScorer:
    # A set of the most common English words for quick matching
    COMMON_WORDS = {
        'THE', 'OF', 'AND', 'A', 'TO', 'IN', 'IS', 'YOU', 'THAT', 'IT', 'HE', 'WAS',
        'FOR', 'ON', 'ARE', 'AS', 'WITH', 'HIS', 'THEY', 'I', 'AT', 'BE', 'THIS',
        'HAVE', 'FROM', 'OR', 'ONE', 'HAD', 'BY', 'WORD', 'BUT', 'NOT', 'WHAT', 'ALL',
        'WERE', 'WE', 'WHEN', 'YOUR', 'CAN', 'SAID', 'THERE', 'USE', 'AN', 'EACH',
        'WHICH', 'SHE', 'DO', 'HOW', 'THEIR', 'IF', 'WILL', 'UP', 'OTHER', 'ABOUT',
        'OUT', 'MANY', 'THEN', 'THEM', 'THESE', 'SO', 'SOME', 'HER', 'WOULD', 'MAKE',
        'LIKE', 'HIM', 'INTO', 'TIME', 'HAS', 'LOOK', 'TWO', 'MORE', 'WRITE', 'GO',
        'SEE', 'NUMBER', 'NO', 'WAY', 'COULD', 'PEOPLE', 'MY', 'THAN', 'FIRST', 'WATER',
        'BEEN', 'CALL', 'WHO', 'OIL', 'ITS', 'NOW', 'FIND', 'LONG', 'DOWN', 'DAY', 'DID',
        'GET', 'COME', 'MADE', 'MAY', 'PART', 'OVER', 'NEW', 'SOUND', 'TAKE', 'ONLY',
        'LITTLE', 'WORK', 'KNOW', 'PLACE', 'YEAR', 'LIVE', 'ME', 'BACK', 'GIVE', 'MOST',
        'VERY', 'AFTER', 'THING', 'OUR', 'JUST', 'NAME', 'GOOD', 'SENTENCE', 'MAN', 'THINK'
    }

    def clean_text(self, text: str) -> str:
        return re.sub(r'[^A-Z ]', '', text.upper())

    def word_score(self, text: str) -> float:
        """Score based on percentage of recognized English words."""
        cleaned = self.clean_text(text)
        words = cleaned.split()
        if not words:
            return 0.0
        
        recognized = sum(1 for word in words if word in self.COMMON_WORDS)
        return recognized / len(words)
