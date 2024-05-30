# MOVIE-GENRE-CLASSIFICATION   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


plot_summaries = movie_data['plot_summary']
genres = movie_data['genre']

plot_summaries_train, plot_summaries_test, genres_train, genres_test = train_test_split(plot_summaries, genres, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()

vectorizer.fit(plot_summaries_train)

plot_summaries_train_tfidf = vectorizer.transform(plot_summaries_train)
plot_summaries_test_tfidf = vectorizer.transform(plot_summaries_test)

classifier = LogisticRegression()

classifier.fit(plot_summaries_train_tfidf, genres_train)

predictions = classifier.predict(plot_summaries_test_tfidf)

print(classification_report(genres_test, predictions))
