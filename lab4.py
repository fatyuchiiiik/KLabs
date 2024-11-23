import os
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random

def analyze_folders(folder_paths):

 lemmatizer = WordNetLemmatizer()
 stop_words = set(stopwords.words('english'))
 k = 2 # Пкф для заголовка

 all_title_word_counts = [] 
 all_text_word_counts = [] 
 for i, folder_path in enumerate(folder_paths):
  # print(f"\nАнализирую папку: {folder_path}")

  title_word_counts = defaultdict(lambda: {'count': 0, 'lemma': ''})
  text_word_counts = defaultdict(lambda: {'count': 0, 'lemma': ''})

  for filename in os.listdir(folder_path):
   if filename.endswith(".txt"):
    filepath = os.path.join(folder_path, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
     text = file.read().lower()

     try:
      title, text = text.split('\n\n', 1)
     except ValueError:
      title = ""
      text = text

     for word in re.findall(r'\b[a-zA-Z]+\b', title):
      if word not in stop_words:
       lemma = lemmatizer.lemmatize(word)
       title_word_counts[lemma]['count'] += 1
       title_word_counts[lemma]['lemma'] = lemma

     for word in re.findall(r'\b[a-zA-Z]+\b', text):
      if word not in stop_words:
       lemma = lemmatizer.lemmatize(word)
       text_word_counts[lemma]['count'] += 1
       text_word_counts[lemma]['lemma'] = lemma

  all_title_word_counts.append(title_word_counts)
  all_text_word_counts.append(text_word_counts)

  with open(f"title_words_{i+1}.txt", "w", encoding='utf-8') as file:
      for word, data in sorted(title_word_counts.items(), key=lambda item: item[1]['count'], reverse=True):
        file.write(f"{data['lemma']}: {data['count']}\n")

  with open(f"text_words_{i+1}.txt", "w", encoding='utf-8') as file:
      for word, data in sorted(text_word_counts.items(), key=lambda item: item[1]['count'], reverse=True):
        file.write(f"{data['lemma']}: {data['count']}\n")

 all_test_title_word_counts = []
 all_test_text_word_counts = []
 for i, folder_path in enumerate(folder_paths):
 
  test_files = random.sample(os.listdir(folder_path), 9)
  
  for filename in test_files:
   if filename.endswith(".txt"):
    filepath = os.path.join(folder_path, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
     text = file.read().lower()

     try:
      title, text = text.split('\n\n', 1)
     except ValueError:
      title = ""
      text = text

          # словари тестовой выборки
     test_title_word_counts = defaultdict(lambda: {'count': 0, 'lemma': ''})
     test_text_word_counts = defaultdict(lambda: {'count': 0, 'lemma': ''})

        
     for word in re.findall(r'\b[a-zA-Z]+\b', title):
      if word not in stop_words:
       lemma = lemmatizer.lemmatize(word)
       test_title_word_counts[lemma]['count'] += 1
       test_title_word_counts[lemma]['lemma'] = lemma

     for word in re.findall(r'\b[a-zA-Z]+\b', text):
      if word not in stop_words:
       lemma = lemmatizer.lemmatize(word)
       test_text_word_counts[lemma]['count'] += 1
       test_text_word_counts[lemma]['lemma'] = lemma

   all_test_title_word_counts.append(test_title_word_counts)
   all_test_text_word_counts.append(test_text_word_counts)
  
   with open(f"test_title_words_{os.path.basename(folder_path)}_{filename}.txt", "w", encoding='utf-8') as file:
      for word, data in sorted(test_title_word_counts.items(), key=lambda item: item[1]['count'], reverse=True):
        file.write(f"{data['lemma']}: {data['count']}n")

   with open(f"test_text_words_{os.path.basename(folder_path)}_{filename}.txt", "w", encoding='utf-8') as file:
      for word, data in sorted(test_text_word_counts.items(), key=lambda item: item[1]['count'], reverse=True):
        file.write(f"{data['lemma']}: {data['count']}n")

  print("nКоэффициенты близости:")
  for j, (test_title_word_counts, test_text_word_counts) in enumerate(zip(all_test_title_word_counts, all_test_text_word_counts)):
    print(f"nТекст {j+1}:")  
    for i, (title_words, text_words) in enumerate(zip(all_title_word_counts, all_text_word_counts)):
      kb = 0
      for word in test_title_word_counts:
        if word in title_words:
          kb += title_words[word]['count'] * k
      for word in test_text_word_counts:
        if word in text_words:
          kb += text_words[word]['count']
      print(f"{os.path.basename(folder_paths[i])}: {kb}") 

if __name__ == "__main__":
  folder_paths = [
    r"C:\Users\admin\Desktop\ИС\archive (1)\business",
    r"C:\Users\admin\Desktop\ИС\archive (1)\entertainment",
    r"C:\Users\admin\Desktop\ИС\archive (1)\politics"
  ]
  analyze_folders(folder_paths)

