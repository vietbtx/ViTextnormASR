from .utils import choice

THOUSANDS = {
    "nghìn": 0.8,
    "ngàn": 0.2
}

MILLIONS = {
    "triệu": 1.0
}

BILLIONS = {
    "tỷ": 0.2,
    "tỉ": 0.8
}

ZERO_CONJUNCTIONS = {
    "linh": 0.8,
    "lẻ": 0.2
}

ONE_UNITS = {
    "mốt": 0.9,
    "một": 0.1
}

LAST_TEN_CONJUNCTIONS = {
    "mươi": 0.9,
    "chục": 0.1
}

TEN_CONJUNCTIONS = {
    "mươi": 0.8
}

SEVENS = {
    "bảy": 0.8,
    "bẩy": 0.2
}

LAST_ZEROS = {
    "không": 0.8,
    "mươi": 0.2
}

LAST_ONES = {
    "một": 0.5,
    "mốt": 0.5
}

LAST_FIVES = {
    "năm": 0.5,
    "lăm": 0.5,
    "nhăm": 0.1
}

LAST_FOURS = {
    "bốn": 0.7,
    "tư": 0.3
}

ZERO_HUNDREDS = {
    "không trăm": 0.8
}

DIGITS = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": choice(SEVENS),
    "8": "tám",
    "9": "chín",
}

SHORTSCALES = {
    "hundred": "trăm",
    "thousand": choice(THOUSANDS), 
    "million": choice(MILLIONS), 
    "billion": choice(BILLIONS)
}

PUNCTUATION = {
    ".": "PERIOD",
    ",": "COMMA",
    "?": "QMARK",
    "!": "EMARK",
    ":": "COLON",
    "-": "DASH",
    ";": "SCOLON"
}

SPECIAL_CHARACTERS = {
    "↑": "tăng",
    "↓": "giảm",
    "&": "và",
    "%": "phần trăm",
    "...": "…"
}

BRACKETS = ["(", ")", "[", "]", "{", "}", "<", ">", "'", '"']
WORDS_END_WITH_PERIOD = ["Dr.", "tp.", "Tp.", "TP.", "Bs.", "BS.", "ThS.", "TS.", "mr.", "Mr.", "MR."]

ALPHABET = {
    "A": "a",
    "B": "bê",
    "C": "xê",
    "D": "đê",
    "Đ": "đê",
    "E": "e",
    "F": choice({"ép": 0.8, "ép phờ": 0.2}),
    "G": "gờ",
    "H": choice({"hắt": 0.8, "hát": 0.2}),
    "I": "i",
    "J": "di",
    "K": "ca",
    "L": "lờ",
    "M": choice({"mờ": 0.8, "em mờ": 0.2}),
    "N": choice({"nờ": 0.8, "en nờ": 0.2}),
    "O": choice({"o": 0.8, "ô": 0.2}),
    "P": "pê",
    "Q": "quy",
    "R": "rờ",
    "S": choice({"ét": 0.8, "ét xì": 0.2}),
    "T": "tê",
    "U": "u",
    "V": "vê",
    "X": choice({"ích": 0.5, "ích xì": 0.3, "ách xì": 0.2}),
    "W": choice({"vê kép": 0.3, "đáp bờ liu": 0.1, "đáp bờ niu": 0.1, "đáp liu": 0.1, "đáp niu": 0.1, "vê đúp": 0.3}),
    "Y": "y",
    "Z": "dét",
}

DAY_PREFIXES = ["hôm", "nay", "qua", "ngày", "sáng", "trưa", "chiều", "tối", "đêm", "khuya", "lúc", "tận", "đến"]
HOUR_PREFIXES = ["nay", "qua", "sáng", "trưa", "chiều", "tối", "đêm", "khuya", "lúc", "tận", "đến", "vào", "trước", "sau"]

TEN = "mười"
ZERO_CONJUNCTION = choice(ZERO_CONJUNCTIONS)
LAST_TEN_CONJUNCTION = choice(LAST_TEN_CONJUNCTIONS)
ONE_UNIT = choice(ONE_UNITS)
TEN_CONJUNCTION = choice(TEN_CONJUNCTIONS)
LAST_ZERO = choice(LAST_ZEROS)
LAST_ONE = choice(LAST_ONES)
LAST_FIVE = choice(LAST_FIVES)
LAST_FOUR = choice(LAST_FOURS)
ZERO_HUNDRED = choice(ZERO_HUNDREDS)
DAY = "ngày"
MONTH = "tháng"
YEAR = "năm"

HOUR = "giờ"
MINUTE = "phút"
SECOND = "giây"

DATE_FROM = "từ"
DATE_TO = choice({"tới": 0.4, "đến": 0.3, "cho tới": 0.1, "cho đến": 0.1, "tới tận": 0.1})