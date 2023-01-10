# -- Functions to extract song names from Spotify playlists, and fetch their lyrics with Genius API --

# Transform spotify playlist URL to URI

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


Client_ID = "14e941e227d44c49a3be9349f1976632"
Client_Secret = "3d20ff8d482f4477a499283081bc7e98"


cid = Client_ID
secret = Client_Secret
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

def url_to_uri (x):
    return x[34:].split('?',1)[0]


# Download song names into a dataframe from a given URI

def playlist_to_dataframe(uri):

    name_list = []
    artist_list = []
    id_list = []

    results = sp.playlist_tracks(uri)
    tracks = results['items']

    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    try:
        for each in tracks:
            name_list.append(each['track']['name'])
    except:
        pass

    try:

        for each in tracks:
            artist_name = ", ".join(
                    [artist["name"] for artist in each["track"]["artists"]]
                )
            id = ", ".join(
                    [artist["id"] for artist in each["track"]["artists"]]
                )
            artist_list.append(artist_name)
            id_list.append(id)
    except:
        pass
    

    df_playlist = pd.DataFrame([name_list,artist_list,id_list]).T

    columns = ['song_name','artist','artist_id']

    df_playlist.columns = columns

    df_playlist.drop_duplicates(inplace=True)

    return df_playlist


# Clean the song names by removing parentheses and extra information

def no_feat(x):
    return x.split('(',1)[0]

def no_from(x):
    return x.split('-',1)[0].strip()


# Add a seperate column for the first and second featuring artist

from unidecode import unidecode

def feat_column(x,type):
    x = str(x)
    cut = x.split(',',3)
    if type == 'main':
        return cut[0]
    elif type == 'feat':
        if len(cut) >= 2:
            return cut[1]
        else:
            return np.nan
    elif type == 'feat2':
        if len(cut) >= 3:
            return cut[2]
        else:
            return np.nan

def add_featuring(df):
    df['song_name'] = df['song_name'].apply(no_feat)
    df['song_name'] = df['song_name'].apply(no_from)
    df['main_artist'] = df['artist'].apply(feat_column,type='main')
    df['feat_artist'] = df['artist'].apply(feat_column,type='feat')
    df['feat_artist2'] = df['artist'].apply(feat_column,type='feat2')
    df['main_artist'] = df['main_artist'].str.lower()
    df['feat_artist'] = df['feat_artist'].str.lower()
    df['feat_artist2'] = df['feat_artist2'].str.lower()

    for col in ['main_artist','feat_artist','feat_artist2']:
        df[col] = df[col].apply(str)
        df[col] = df[col].apply(unidecode)
        df[col] = df[col].replace('nan',np.NaN)

    df.drop_duplicates(inplace=True)
    df = df.drop('artist',axis=1)
    return df


# ---------------------------------------------------
# Download the gender CSV and cross reference it with our dataframe on the common artist names


gender_csv1 = pd.read_csv('top_artist_gender.csv')

gender_csv1 = gender_csv1[['artist','gender']]
gender_csv1['artist'] = gender_csv1['artist'].str.lower()
gender_csv1['artist'] = gender_csv1['artist'].apply(unidecode)

gender_csv2 = pd.read_csv('Gender_Dataset_2.csv',low_memory=False)
gender_csv2 = gender_csv2[['name','gender']]
columns_csv2 = ['artist','gender']
gender_csv2.columns = columns_csv2
gender_csv2 = gender_csv2[~gender_csv2['gender'].isnull()]
gender_csv2['artist'] = gender_csv2['artist'].apply(unidecode).str.lower()

gender_csv = pd.concat([gender_csv1,gender_csv2])
gender_csv.drop_duplicates(inplace=True)




def merge_artists(df):
    uncommon = pd.merge(gender_csv,df,how='inner', left_on ='artist', right_on = 'main_artist')
    uncommon['feat_artist'] = uncommon['feat_artist'].str.strip()
    gender_series = gender_csv[['artist','gender']]
    gender_series.drop_duplicates(inplace=True)

    feat = pd.merge(uncommon,gender_series,how='left',left_on='feat_artist',right_on='artist')

    feat_cols = ['artist_x','gender_main','song_name','artist_id','main_artist','feat_artist','feat_artist2','artist','gender_feat']

    feat.columns = feat_cols
    feat = feat.drop('artist',axis=1)

    feat['feat_artist2'] = feat['feat_artist2'].str.strip()

    total_feat = pd.merge(feat,gender_series,how='left',left_on='feat_artist2',right_on='artist')

    total_feat = total_feat.drop('artist',axis=1)

    feat_cols2 = ['artist_x','gender_main','song_name','artist_id','main_artist','feat_artist','feat_artist2','gender_feat','gender_feat2']

    total_feat.columns = feat_cols2

    total_feat.drop_duplicates(inplace=True)

    total_feat = total_feat.drop('artist_x',axis=1)

    return total_feat


# ---------------------------------------------------
# Create a function to find the songs with similarly gendered artists, and keep only those in a dataframe

# *** To be used in the next function ***

def same_gender(main,feat,feat2):
    x = list((main,feat,feat2))
    if 'male' in x and 'female' in x:
        return 0
    elif 'mixed' in x or 'other' in x:
        return 0
    else:
        return 1


def same_gender_df(df):
    df = df[(df['gender_main']=='male') | (df['gender_main'] =='female')]

    df['same_gender'] = df.apply(lambda x: same_gender(x.gender_main,x.gender_feat,x.gender_feat2),axis=1)

    df_samegender = df.loc[df['same_gender']==1]

    return df_samegender


def final_gender_df(df):
    full_gender = df[(df['gender_main'] != 'other') & (df['same_gender'] == 1)]
    full_gender = full_gender.drop_duplicates(subset=['song_name','main_artist'],keep='last').reset_index(drop=True)
    return full_gender


# ---------------------------------------------------
# Load the Genius API and tokens to fetch lyrics

from lyricsgenius import Genius
from requests.exceptions import Timeout


genius_client_id = "DtPAMtYR8AtXFDiRJp62FbDsIlz6kLWnU8M0na5WNRU2mLJy329CHvTHXipsoSpz"

genius_secret = "P8YoSMZTCRlUTQeWwkDAwpzcw1x6YoY2gov5z1ToFGftckYf5AkT4ygANw6_E0-UhLoJrLOV9bM4jjTa6ZH6mQ"

token = 'hu-wUFM2nfgnCpkohDZztJ7c8XscFHiYous3dtCdGMQGcCKtsPnXx2k82eDyZHmq'

genius = Genius(token)


# ---------------------------------------------------
# Fetch the lyrics from a given dataframe using the artist name and the song name with the Genius API


def fetch_lyrics(df):
    genius = Genius(token)

    song_list = df['song_name'].tolist()
    
    artist_list = df['main_artist'].tolist()

    lyric = []
    for i in range(len(song_list)):
        retries = 0
        while retries < 3:
            try:
                song = genius.search_song(song_list[i],artist_list[i])
            except Timeout as e:
                retries += 1
                continue
            if song is not None:
                lyric.append(song.lyrics)
            else:
                lyric.append('No Lyrics')
            break
    return [str(x).lower() for x in lyric]



# ---------------------------------------------------
# Cleanup the raw lyrics fetched from the Genius API

def no_bracket(x_full):
    rep = []
    for lyric in x_full:
        lyric = str(lyric)
        ret = ''
        skip1c = 0
        for i in lyric:
            if i == '[':
                skip1c += 1
            elif i == ']' and skip1c > 0:
                skip1c -= 1
            elif skip1c == 0:
                ret += i
        rep.append(ret)
    return rep

def no_embe(x_full):
    new_lis = []
    for x in x_full:
        takeout = x[-8:]
        takeout = takeout.replace('embed','')
        end = ""
        for letter in takeout:
            if not letter.isdigit():
                end = str (end + letter)
        new_lis.append(str(x[:-8] + end))
    return new_lis

def strip_split(x_full):
    strip_split = []
    for lyric in x_full:
        strip_split.append(lyric.strip("\u200b").split('lyrics',1)[1])
    return strip_split


def replace(x_full):
    new = []
    to_replace = ["\n","\\","\u200b","\u2005","\u205f","\u200a"]
    for x in x_full:
        for i in to_replace:
            x = x.replace(i,' ')
        x.replace("\\'","'")
        new.append(x)
    return new
    

def no_ticket(x_full):
    rep = []
    for x in x_full:
        index_debut=0
        index_fin=0
        y = x.split()
        if 'liveget' in y:
            for i in range(len(y)):
                if y[i] == 'liveget':
                    if 'see' in y[i-2]:
                        index_debut = i-2
                    elif 'see' in y[i-3]:
                        index_debut = i-3
                    elif 'see' in y[i-4]:
                        index_debut = i-4
                    index_fin = i +9
            del y[index_debut:index_fin]
            y_string = ''.join(str(e)+' ' for e in y)
            rep.append(y_string)
        else:
            rep.append(x)
    return rep


def chain(start, *funcs):
    res = start
    for func in funcs:
        res = func(res)
    return res


def full_cleanup(x_full):
    x_full = [str(x) for x in x_full]
    x_cleaned = chain(x_full,no_bracket,no_embe,strip_split,replace,no_ticket)
    return x_cleaned


# ---------------------------------------------------
#Chain of functions to fetch the songs and lyrics

def fetch_song_dataframe(url):
    gender_df = chain(url,url_to_uri,playlist_to_dataframe,add_featuring,merge_artists)
    gender_df_final = chain(gender_df,same_gender_df,final_gender_df)
    return gender_df_final

def fetch_lyrics_list(song_dataframe):
    return chain(song_dataframe,fetch_lyrics,full_cleanup)


#Apply the predicted values to the dataframe

def to_final_df(list_of_lyrics,gender_df):
    lyrics = pd.Series(list_of_lyrics)
    full_df = gender_df.assign(lyrics = lyrics)
    df_predict = full_df[['gender_main','lyrics']]
    df_predict['gender'] = df_predict['gender_main'].apply(lambda x : 1 if x == 'male' else 0)
    df_predict = df_predict.drop('gender_main',axis=1)
    df_predict = df_predict.loc[df_predict['lyrics'] != '']

    df_predict.drop_duplicates(inplace=True)
    return df_predict

# Remove any numbers and other exceptions

def has_numbers(inp):
    return any(char.isdigit() for char in inp)

def isEnglish(x):
    empty = []
    x = x.split()
    for s in x:
        if has_numbers(s) == False:
            try:
                s.encode(encoding='utf-8').decode('ascii')
            except UnicodeDecodeError:
                pass
            else:
                empty.append(s)
    return (empty)



#Plot the proportion of male and female artists in our dataset

def gender_plot(df):
    plt.figure(figsize=(4,5))
    sns.countplot(data =df, x='gender_main',palette='winter')



# -- Functions used for model fititng --

from sklearn.metrics import confusion_matrix,classification_report,recall_score,precision_score,accuracy_score
import pandas as pd
import numpy as np


def model_score(model,x_test,x_train,y_test,y_train):
	preds_test = model.predict(x_test)
	preds_train = model.predict(x_train)
	cmat = confusion_matrix(y_test, preds_test)
	print(f'Train Accuracy: {accuracy_score(y_train, preds_train)}')
	print(f'Test Accuracy: {accuracy_score(y_test, preds_test)}')
	print(classification_report(y_test, preds_test))
	return pd.DataFrame(cmat, columns=['Predicted ' + str(i) for i in ['Women','Men']],index=['Actual ' + str(i) for i in ['Women','Men']])


def nn_model_score(model,x_test,x_train,y_test,y_train):
    print('Train Results:')
    print(model.evaluate(x_train,np.asarray(y_train)))
    print('Test Results:')
    print(model.evaluate(x_test,np.asarray(y_test)))



