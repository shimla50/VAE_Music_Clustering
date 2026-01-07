import pandas as pd # type: ignore

df = pd.read_csv("spotify_songs.csv")  # Ensure the CSV file is in the correct directory or provide full path
df.head()
