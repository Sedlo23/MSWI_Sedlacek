from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

# Načtení datasetu z CSV souboru
data = pd.read_csv("twitter_MBTI.csv")

# Vypočítání počtu tweetů pro každý MBTI typ
type_counts = data["label"].value_counts()

# Vytvoření sloupcového grafu pro distribuci MBTI typů
plt.figure(figsize=(8,6))
plt.bar(type_counts.index, type_counts.values)
plt.title("Distribution of MBTI types")
plt.xlabel("MBTI type")
plt.ylabel("Count")
plt.show()

# Rozdělení dat podle MBTI typu
type_groups = data.groupby("label")

# Vytvoření obrázku pro wordclouds pro každý MBTI typ
fig = plt.figure(figsize=(12, 10))

sentiment_data = []
for name, group in type_groups:
    # Získání tweetů pro daný MBTI typ
    tweets = group["text"].tolist()
    polarity_scores = []
    # Výpočet polarity pro každý tweet pomocí TextBlob
    for tweet in tweets:
        polarity_scores.append(TextBlob(tweet).sentiment.polarity)
    sentiment_data.append((name, polarity_scores))


# Cyklus pro přidání grafů WordCloud do jednoho obrázku pro každý MBTI typ
for i, (name, group) in enumerate(type_groups):
    tweets = group["text"].tolist()
    words = []
    for tweet in tweets:
        # Filtrace slov kratších než 10 znaků
        new_words = list(filter(lambda x: len(x) >= 10, tweet.split()))
        words.extend(new_words)
    word_counts = Counter(words)
    wc = WordCloud(background_color="white", max_words=100, width=400, height=200)
    wc.generate_from_frequencies(word_counts)
    ax = fig.add_subplot(4, 4, i+1)
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(f"{name}")
    ax.axis("off")

# Nastavení obecného titulu pro celý obrázek
fig.suptitle("Word clouds for MBTI personality types")

plt.show()

# Vytvoření sloupcového grafu pro analýzu sentimentu podle MBTI typu
plt.figure(figsize=(12,6))
for i, (name, scores) in enumerate(sentiment_data):
    plt.bar(i, sum(scores)/len(scores), label=name)
plt.xticks(range(len(sentiment_data)), [name for name, _ in sentiment_data])
plt.title("Sentiment analysis by MBTI type")
plt.xlabel("MBTI type")
plt.ylabel("Average polarity score")
plt.legend()
plt.show()
