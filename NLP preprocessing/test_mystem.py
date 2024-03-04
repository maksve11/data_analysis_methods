from pymystem3 import Mystem
from collections import Counter

# Загрузка текста произведения "Вий" Н.В. Гоголя из файла
with open('gogol.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Инициализация Mystem
mystem = Mystem()

# Получение нормальной формы слов в тексте
normalized_words = [word.strip().lower() for word in mystem.lemmatize(text) if word.isalpha()]

# Подсчет частоты каждого слова
word_frequencies = Counter(normalized_words)

# Вывод частот указанных слов
print("Частота слова 'бурса':", word_frequencies['бурса'])
print("Частота слова 'собака':", word_frequencies['собака'])
print("Частота слова 'чувство':", word_frequencies['чувство'])
print("Частота слова 'малороссиянин':", word_frequencies['малороссиянин'])
