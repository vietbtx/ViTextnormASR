
import tools
import importlib

class Numb2Text:
    
    def reset():
        importlib.reload(tools)

    def read(text: str, allow_alphabet: bool = False):
        text = Numb2Text.process_space(text)
        text = Numb2Text.read_date(text)
        text = Numb2Text.read_time(text)
        data = Numb2Text.read_number(text, allow_alphabet)
        result = []
        for text, label in data:
            if label == "O":
                result += Numb2Text.read_cap(text)
            else:
                if label == "PUNC":
                    if text in tools.PUNCTUATION:
                        text = tools.PUNCTUATION[text]
                    else:
                        label = "O"
                result += [[text, label]]
        return result
    
    def process_space(text: str):
        text = " " + text + " "
        for c, word in tools.SPECIAL_CHARACTERS.items():
            text = text.replace(f"{c}", f" {word} ")
        for c in tools.BRACKETS:
            text = text.replace(f"{c}", f" {c} ")
        for c in tools.PUNCTUATION:
            text = text.replace(f"{c} ", f" {c} ")
            text = text.replace(f" {c}", f" {c} ")
        text = " ".join(text.split())
        text = " " + text + " "
        for word in tools.WORDS_END_WITH_PERIOD:
            text = text.replace(f" {word[:-1]} . ", f" {word} ")
        return text

    def read_cap(text: str):
        text = Numb2Text.process_space(text)
        result = []
        words = text.split()
        for word in words:
            if word[0].isalpha() and word[0].isupper():
                if all(c.isalpha() and c.isupper() and c in tools.ALPHABET for c in word):
                    result.append([Numb2Text.read_characters_only(word, True), "ALLCAP"])
                else:
                    result.append([word.lower(), "CAP"])
            else:
                tools.utils.append_or_add(result, word.lower(), "O")
        return result        
    
    def check_day(day: str):
        day = tools.utils.is_number(day)
        if day is not None and 0 < day <= 31:
            return day

    def check_month(month: str):
        month = tools.utils.is_number(month)
        if month is not None and 0 < month <= 12:
            return month
    
    def check_year(year: str):
        year = tools.utils.is_number(year)
        if year is not None and 1900 <= year <= 9999:
            return year

    def check_full_date(day: str, month: str, year: str):
        day = Numb2Text.check_day(day)
        month = Numb2Text.check_month(month)
        year = Numb2Text.check_year(year)
        if day and month and year:
            return f"{day} {tools.MONTH} {month} {tools.YEAR} {year}"
    
    def check_short_date_type_1(day: str, month: str):
        day = Numb2Text.check_day(day)
        month = Numb2Text.check_month(month)
        if day and month:
            return f"{day} {tools.MONTH} {month}"
    
    def check_short_date_type_2(month: str, year: str):
        month = Numb2Text.check_month(month)
        year = Numb2Text.check_year(year)
        if month and year:
            return f"{month} {tools.YEAR} {year}"
    
    def check_range_date_type_1(start: str, end: str, month: str):
        start = Numb2Text.check_day(start)
        end = Numb2Text.check_day(end)
        month = Numb2Text.check_month(month)
        if start and end and month and start < end:
            return f"{start} {tools.DATE_TO} {end} {tools.MONTH} {month}"
    
    def check_range_date_type_2(start: str, end: str, year: str):
        start = Numb2Text.check_month(start)
        end = Numb2Text.check_month(end)
        year = Numb2Text.check_year(year)
        if start and end and year and start < end:
            return f"{start} {tools.DATE_TO} {tools.MONTH} {end} {tools.YEAR} {year}"

    def check_date(word: str, p_word: str=None):
        parts = Numb2Text.split_numb(word)
        date = None
        if len(parts) == 5:
            if parts[1] == parts[3] and parts[1] in ".-/\\":
                date = Numb2Text.check_full_date(parts[0], parts[2], parts[4])
            if not date and parts[1] == "-" and parts[3] in ".-/\\":
                date = Numb2Text.check_range_date_type_1(parts[0], parts[2], parts[4])
                if not date:
                    date = Numb2Text.check_range_date_type_2(parts[0], parts[2], parts[4])
                    if date and p_word != tools.MONTH:
                        date = f"{tools.MONTH} {date}"
        elif len(parts) == 3:
            if p_word and p_word in tools.DAY_PREFIXES and parts[1] == "/":
                date = Numb2Text.check_short_date_type_1(parts[0], parts[2])
            if not date and parts[1] in ".-/\\":
                date = Numb2Text.check_short_date_type_2(parts[0], parts[2])
                if date and p_word != tools.MONTH:
                    date = f"{tools.MONTH} {date}"
        return date

    def read_date(text: str):
        words = text.split()
        result = []
        for k, word in enumerate(words):
            p_word = words[k-1].lower() if k>0 else None
            date = Numb2Text.check_date(word, p_word)
            if not date:
                parts = word.split("-")
                if len(parts) == 2:
                    date1 = Numb2Text.check_date(parts[0], tools.DAY_PREFIXES[0])
                    if not date1:
                        date1 = Numb2Text.check_day(parts[0])
                    if date1:
                        date2 = Numb2Text.check_date(parts[1], tools.DAY_PREFIXES[0])
                        if date2:
                            date = f"{date1} {tools.DATE_TO} {date2}"
            result.append(date if date else word)
        return " ".join(result)
    
    def check_full_time(hour: str, minute: str):
        hour = tools.utils.is_number(hour, float)
        minute = tools.utils.is_number(minute)
        if hour and minute and hour > 0 and 0 <= minute < 60:
            return f"{hour} {tools.HOUR} {minute} {tools.MINUTE}"
    
    def check_short_time(numb: str, unit: str):
        numb = tools.utils.is_number(numb, float)
        if numb and numb > 0:
            if unit in "hg":
                return f"{numb} {tools.HOUR}"
            elif unit in ["m", "p", "ph"]:
                return f"{numb} {tools.MINUTE}"
            elif unit in "s":
                return f"{numb} {tools.SECOND}"
    
    def check_time(word: str, p_word: str=None):
        parts = Numb2Text.split_numb(word, ".,")
        date = None
        if len(parts) == 2:
            if p_word and p_word in tools.HOUR_PREFIXES:
                date = Numb2Text.check_short_time(parts[0], parts[1])
        elif len(parts) == 3:
            minute = tools.utils.is_number(parts[2])
            if minute and 0 <= minute < 60:
                date = Numb2Text.check_short_time(parts[0], parts[1])
                if date:
                    date = f"{date} {minute}"
        elif len(parts) == 4:
            if parts[1] in "hg" and parts[3] in ["m", "p", "ph"]:
                date = Numb2Text.check_full_time(parts[0], parts[2])
        return date

    def read_time(text: str):
        words = text.split()
        result = []
        for k, word in enumerate(words):
            p_word = words[k-1].lower() if k>0 else None
            time = Numb2Text.check_time(word, p_word)
            result.append(time if time else word)
        return " ".join(result)

    def read_number(text: str, allow_alphabet: bool = False):
        result = []
        for word in text.split():
            if word not in tools.PUNCTUATION:
                data = Numb2Text._read_number(word, allow_alphabet)
            else:
                data = [[word, "PUNC"]]
            for word, label in data:
                tools.utils.append_or_add(result, word, label)
        return result

    def _read_number(text: str, allow_alphabet: bool = False):
        parts = Numb2Text.split_numb(text)
        parts = Numb2Text.split_thousand_unit(parts, ",", ".")
        if "," in parts:
            parts = Numb2Text.split_thousand_unit(parts, ".", ",")
        result = []
        is_first_digit = True
        is_correct_numb = True
        for part in parts:
            if part.isdigit():
                if is_first_digit and part[0] == "0" or not is_correct_numb:
                    result.append([Numb2Text.read_characters_only(part), "NUMB"])
                else:
                    result.append([Numb2Text.read_digits(part), "NUMB"])
                is_first_digit = False
            elif part in tools.SHORTSCALES:
                if len(result) > 0 and len(result[-1][0]) > 0:
                    result.append([tools.SHORTSCALES[part], "NUMB"])
            else:
                tools.utils.append_or_add(result, Numb2Text.read_characters_only(part, allow_alphabet), "O")
                if part in ".,":
                    is_correct_numb = False
        return result

    def split_numb(text: str, allowed_chars: list = []):
        isdigit = lambda c: c.isdigit() or c in allowed_chars
        parts = []
        for c in text:
            if len(parts) == 0 or isdigit(c) != is_digit:
                parts.append(c)
            else:
                parts[-1] += c
            is_digit = isdigit(c)
        return parts

    def split_thousand_unit(parts: list, decimal_char: str = ".", split_char: str = ","):
        decimal_index = tools.utils.index(parts, decimal_char)
        integral = parts[:decimal_index]
        fractional = parts[decimal_index:]
        scale_units = ["thousand", "million", "billion"]
        if tools.utils.is_correct_fractional(fractional):
            if len(integral) == 1:
                integral = integral[0]
                if integral[0] != "0" and integral.isdigit():
                    del parts[decimal_index - 1]
                    i = 0
                    while len(integral) > 0:
                        part = integral[-3:]
                        integral = integral[:-3]
                        parts.insert(decimal_index - 1, part)
                        parts.insert(decimal_index - 1, scale_units[i % 3])
                        i += 1
                    del parts[decimal_index - 1]
            if tools.utils.is_correct_integral(integral, split_char):
                i = 1
                while i * 2 < len(integral):
                    parts[decimal_index - i * 2] = scale_units[(i - 1) % 3]
                    i += 1
        return parts

    def _read_1_digit(text: str):
        assert len(text) == 1
        return tools.DIGITS.get(text)
    
    def _read_first_digit(c1: str, c2: str):
        part1 = ""
        if c1 == "0":
            if c2 != "0":
                part1 = tools.ZERO_CONJUNCTION
        elif c1 == "1":
            part1 = tools.TEN
        else:
            part1 = Numb2Text._read_1_digit(c1)
            if c2 != "0":
                part1 = f"{part1} {tools.TEN_CONJUNCTION}".strip()
        return part1
    
    def _read_second_digit(c1: str, c2: str):
        part2 = ""
        if c2 == "0":
            if c1 not in "01":
                part2 = tools.LAST_TEN_CONJUNCTION
        elif c2 == "1" and c1 not in "01":
            part2 = tools.ONE_UNIT
        else:
            if c2 == "5" and c1 != "0":
                part2 = tools.LAST_FIVE
            elif c2 == "4" and c1 not in "01":
                part2 = tools.LAST_FOUR
            else:
                part2 = Numb2Text._read_1_digit(c2)
            
        return part2

    def _read_2_digits(text: str):
        assert len(text) == 2
        c1, c2 = text
        part1 = Numb2Text._read_first_digit(c1, c2)
        part2 = Numb2Text._read_second_digit(c1, c2)
        return f"{part1} {part2}".strip()

    def _read_3_digits(text: str):
        assert len(text) <= 3
        if len(text) == 1:
            return Numb2Text._read_1_digit(text)
        elif len(text) == 2:
            return Numb2Text._read_2_digits(text)
        c1 = text[0]
        c2 = text[1:]
        if c1 == "0":
            part1 = tools.ZERO_HUNDRED if c2 != "00" else ""
        else:
            part1 = Numb2Text._read_1_digit(c1)
            part1 = f"{part1} {tools.SHORTSCALES['hundred']}"
        part2 = Numb2Text._read_2_digits(c2)
        return f"{part1} {part2}".strip()
    
    def _read_6_digits(text: str):
        assert len(text) <= 6
        if len(text) <= 3:
            return Numb2Text._read_3_digits(text)
        c1 = text[:-3]
        c2 = text[-3:]
        part1 = Numb2Text._read_3_digits(c1)
        if part1 != tools.DIGITS["0"] and len(part1) > 0:
            part1 = f"{part1} {tools.SHORTSCALES['thousand']}" 
        else:
            part1 == ""
        part2 = Numb2Text._read_3_digits(c2)
        return f"{part1} {part2}".strip()
    
    def _read_9_digits(text: str):
        assert len(text) <= 9
        if len(text) <= 6:
            return Numb2Text._read_6_digits(text)
        c1 = text[:-6]
        c2 = text[-6:]
        part1 = Numb2Text._read_3_digits(c1)
        if part1 != tools.DIGITS["0"] and len(part1) > 0:
            part1 = f"{part1} {tools.SHORTSCALES['million']}" 
        else:
            part1 == ""
        part2 = Numb2Text._read_6_digits(c2)
        return f"{part1} {part2}".strip()
    
    def read_digits(text: str):
        result = ""
        is_start = True
        while len(text) > 0:
            word = Numb2Text._read_9_digits(text[-9:])
            if is_start:
                result = word
                is_start = False
            else:
                if len(word) == 0:
                    word = tools.DIGITS["0"]
                result = f"{word} {tools.SHORTSCALES['billion']} {result}".strip()
            text = text[:-9]
        return result
    
    def read_characters_only(text: str, allow_alphabet: bool = False):
        result = []
        is_valid = False
        for k, c in enumerate(text):
            if c in tools.DIGITS:
                is_correct = False
                if k+1 >= len(text) or not text[k+1].isdigit():
                    if k > 0 and text[k-1].isdigit() and text[k-1] not in "01":
                        if c == "0":
                            result.append(tools.LAST_ZERO)
                            is_correct = True
                        elif c == "1":
                            result.append(tools.LAST_ONE)
                            is_correct = True
                        elif c == "4":
                            result.append(tools.LAST_FOUR)
                            is_correct = True
                if not is_correct:
                    result.append(tools.DIGITS[c])
            elif c in tools.PUNCTUATION:
                result.append(c)
            elif allow_alphabet and c.upper() in tools.ALPHABET:
                result.append(tools.ALPHABET[c.upper()])
            else:
                if not is_valid and len(result) > 0:
                    result[-1] += c
                else:
                    result.append(c)
                is_valid = False
            if is_valid:
                is_valid = False
        result = " ".join(word for word in result if len(word) > 0)
        return result