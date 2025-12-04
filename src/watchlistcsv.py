import pandas as pd
import glob
import re

movies = pd.read_csv('movies_tmdb.csv')
movies['title_clean'] = movies['title'].str.strip()

all_rows = []

for file in glob.glob('watchlist_*.csv'):
    user_id = int(re.findall(r"watchlist_(\d+)", file)[0])
    w = pd.read_csv(file)
    
    w['combined'] = w['Name'] + " (" + w['Year'].astype(str) + ")"
    merged = w.merge(movies[['movieId','title_clean']],
                     left_on='combined', right_on='title_clean', how='left')
    
    merged['userId'] = user_id
    all_rows.append(merged[['userId','movieId']])

final = pd.concat(all_rows)
# Eşleşmeyen filmleri (NaN) çıkar
final = final.dropna(subset=['movieId'])
# MovieId'yi integer'a çevir
final['movieId'] = final['movieId'].astype(int)
final.to_csv('watchlist.csv', index=False)
