#py - m streamlit run presentation_sl.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.subplots as sp
from wordcloud import WordCloud, STOPWORDS 
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import streamlit as st

st.set_page_config(layout='centered')

#to make a presentation import streamlit
billboard_df=pd.read_csv('Hot 100 Audio Features.csv')


st.header('Billboard Hot weekly charts')
st.subheader('Music analysis from spotify data')

st.title('DataSet Link - Description')
curtain=st.selectbox('Click to see',('Link','Description of the Dataset'))
if curtain=='Link':
    st.write('https://www.kaggle.com/datasets/thedevastator/billboard-hot-100-audio-features')
if curtain=='Description of the Dataset':
    st.markdown("""
    * Performer:	The name of the performer or artist of the song. (Text)
    * Song:	The title of the song. (Text)
    * spotify_genre:	The genre(s) of the song according to Spotify's classification system. (Text)
    * spotify_track_preview_url:	The URL linking to a preview version of the song on Spotify, if available. (Text)
    * spotify_track_duration_ms:	The duration of the song in milliseconds. (Numeric)
    * spotify_track_explicit:	Indicates whether the song contains explicit content or not. (Boolean)
    * spotify_track_album:	The title of the album associated with the song. (Text)
    * danceability:	A measurement criteria combining musical elements in terms of suitability for dancing. (Numeric)
    * energy:	Characterizes the intensity and activity within each recording. (Numeric)
    * key:	Showcases tonalities such as C major or D minor described using text representations. (Text)
    * loudness:	Expressed as decibel levels measuring the overall volume across songs. (Numeric)
    * mode:	Distinguishes major or minor tonalities indicated as numeric values within this context. (Numeric)
    * speechiness:	Shows the presence of spoken-word elements quantitatively within tracks. (Numeric)
    * acousticness:	Displays the acoustic qualities of songs numerically, reflecting the contrast between natural sound and electronically enhanced production or elements. (Numeric)
    * instrumentalness:	Quantifies the likelihood of a song being instrumental using numeric metrics. (Numeric)
    * liveness:	Reveals the presence of an audience within live recordings via numerical evaluation. (Numeric)
    * valence:	Describes the musical positivity conveyed by songs quantitatively as numeric measurements on a scale. (Numeric)
    * tempo:	Measures the beats per minute (BPM) for tempo indication evaluated by numeric means. (Numeric)
    * time_signature:	Specifies the song structure like 4/4 or 3/4 using text representation. (Text)
    * spotify_track_popularity:	Gauges the popularity of the song on Spotify. (Numeric)            
    """
    )

if st.checkbox('Dataset'):
    st.write(billboard_df.head(5))
    st.write(billboard_df.tail(5))
    st.write(billboard_df.describe().T)

#cleaning df
del billboard_df['spotify_track_id']
del billboard_df['spotify_track_preview_url']

billboard_df=billboard_df[billboard_df['spotify_genre'].notna()]

billboard_df.spotify_track_album.fillna('single',inplace=True)

billboard_df.spotify_track_explicit.fillna('unknown',inplace=True)

lista_main_generi=['pop','metal','rock','rap','trap','jazz','blues','country','indie','house','hip hop','techno','folk','reggae','punk','disco','classical','gospel','soul','instrumental','celtic','drill','dance','r&b','sertanejo','cumbia','electronic','lo-fi','ska','doom','prog','piano','soundtrack','salsa','cumbia','orchestra','beat','psych','alt','emo','old school','romantic','funk','choral','songwriter','talent show','latin','experimental','rumba','glitch','ambient','other']
billboard_df['maingenere']='other'
for index,generi in billboard_df.spotify_genre.items():
  prova=str(generi).replace('[',"").replace(']',"").replace("'","").replace('"',"").replace(',',"").replace('-'," ")
  counts = dict()
  prova2 = prova.split( )
  for word in prova2:
      if word in counts:
          counts[word] += 1
      else:
          counts[word] = 1
  lista_tuple_generi = sorted(counts.items(), key = lambda x:x[1], reverse=True)

  primo = False
  for tuple_genere in lista_tuple_generi:
      if primo == False:
          if tuple_genere[0] in lista_main_generi:
              primo =True

              billboard_df.loc[index,'maingenere']=tuple_genere[0]

for null_column in billboard_df:
    if billboard_df[null_column].isnull().any() == True :
      
        for generi in lista_main_generi:
          
            mask_generi=billboard_df.maingenere==generi
            genere_df=billboard_df[mask_generi]

            mean_generi=round(genere_df[null_column].mean(),2)

            billboard_df.loc[mask_generi, null_column] = billboard_df.loc[mask_generi, [null_column]].fillna(mean_generi)

billboard_df=billboard_df.dropna()

### CORRELATION

if st.checkbox('Preliminary correlation analysis'):
    numeric_variable_df=billboard_df.iloc[:,8:21]
    corr_billboard=numeric_variable_df.corr(method='pearson')
    fig=plt.figure(figsize=(10,8))
    sb.heatmap(corr_billboard,annot = True, fmt = '.1g', vmin=-1, vmax=1, center=0, cmap='Greens', linewidths=0.1, linecolor='black')
    st.write(fig)

    st.markdown("""
    ### We can see that the variable the have the **high correlation** are:
    * 1. Loudnes - energy: 0.7
    * 2. Acousticness - energy: -0.6
    * 3. Loudnes - acousticness: -0.4
    * 4. Loudnes - spotify_track_popularity: 0.4
    * 5. Danceability - valence: 0.4
                """
                )

    fig_1 = plt.figure(figsize=(20, 16))
    ax_1 = fig_1.add_subplot(2, 2, 1)
    ax_2 = fig_1.add_subplot(2, 2, 2)


    ax_1.scatter(billboard_df.energy,billboard_df.loudness)
    ax_1.title.set_text('Relation between Energy - Loudness')

    ax_2.scatter(billboard_df.energy,billboard_df.acousticness)
    ax_2.title.set_text('Relation between Energy - Acusticnes')

    st.write(fig_1)

#12 maingeneri
numbers_of_genere=billboard_df.maingenere.value_counts()[0:12]

st.title('What are the main genres?')

if st.checkbox('Distribution of Main generes'):
    fig=plt.figure(figsize=(12,10))
    color=sb.color_palette('pastel6')
    explode=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    lables=numbers_of_genere.index
    plt.pie(numbers_of_genere,labels=lables, startangle=25,labeldistance=0.8,autopct='%1.0f%%',colors=color,explode=explode)
    plt.title('Percentage of songs for genre',size=22,color='black')
    plt.legend()
    st.write(fig)

if st.checkbox('12 main genres presence'):
    genere_distribution=billboard_df.maingenere.value_counts()
    genere_distribution_no_col_maingenere=genere_distribution.to_string(header=False)

    genere_importance = WordCloud(width=800, height=600, margin=0).generate(str(genere_distribution_no_col_maingenere))

    fig=plt.figure(figsize=(12,8))
    plt.imshow(genere_importance, interpolation='bilinear')

    plt.axis("off")
    #plt.show()
    st.pyplot(fig)


numbers_of_genere=billboard_df.maingenere.value_counts()[0:12]

worst_genere=billboard_df.maingenere.value_counts()[12:]

top_12_genere_df=billboard_df.copy()

for i in worst_genere.index:
    top_12_genere_df=top_12_genere_df.drop(top_12_genere_df[top_12_genere_df.maingenere==i].index)

top_12_genere_df.spotify_track_duration_ms=round(top_12_genere_df.spotify_track_duration_ms/60000,2)
top_12_genere_df.rename(columns={'spotify_track_duration_ms':'spotify_track_duration_minute'},inplace=True)

top_12_genere_gb=top_12_genere_df.groupby('maingenere').mean(numeric_only=True)

explicit_songs = top_12_genere_df[top_12_genere_df.spotify_track_explicit == True]
explicit_by_genre = explicit_songs['maingenere'].value_counts()

not_explicit_songs = top_12_genere_df[top_12_genere_df.spotify_track_explicit == False]
not_explicit_by_genre = not_explicit_songs['maingenere'].value_counts()

unknown_explicit_songs = top_12_genere_df[top_12_genere_df.spotify_track_explicit == 'unknown']
unknown_explicit_by_genre = unknown_explicit_songs['maingenere'].value_counts()

explicit_df=pd.concat([explicit_by_genre,not_explicit_by_genre,unknown_explicit_by_genre],axis=1)
explicit_df.columns=['explicit', 'not_explicit','unknown_explicit']
explicit_df['maingenere'] = explicit_df.index

dist_explicit=top_12_genere_df.spotify_track_explicit.value_counts()

curtain1=st.selectbox('Click to see - characteristics for maingenre',('None','Popularity','Duration','Danceability, Energy, Valence',
                                      'Speechiness, Instrumentalness, Liveness, Acousticness','Explicit content','BPM'))

if curtain1 == 'None':
    pass
if curtain1=='Popularity':
    plt.figure(figsize=(12,8))
    fig4 = px.bar(top_12_genere_gb, x=top_12_genere_gb.index, y=['spotify_track_popularity'], barmode='group',text_auto=True)
    fig4.update_layout(title=dict(text="What is the most popular genre?", font=dict(size=40), automargin=True, yref='paper'))
    st.write(fig4)
if curtain1=='Duration':
    fig3 = px.bar(top_12_genere_gb, x=top_12_genere_gb.index, y=['spotify_track_duration_minute'], barmode='group',text_auto=True)
    fig3.update_layout(title=dict(text="Duration in minute for genere", font=dict(size=40), automargin=True, yref='paper'))
    st.write(fig3)
if curtain1=='Danceability, Energy, Valence':
    fig1 = px.bar(top_12_genere_gb, x=top_12_genere_gb.index, y=['danceability','energy','valence'], barmode='group',text_auto=True)
    fig1.update_layout(title=dict(text="Main characteristics for genere", font=dict(size=40), automargin=True, yref='paper'))
    st.write(fig1)
if curtain1=='Speechiness, Instrumentalness, Liveness, Acousticness':
    fig2 = px.bar(top_12_genere_gb, x=top_12_genere_gb.index, y=['speechiness','instrumentalness','liveness','acousticness'], barmode='group',text_auto=True)
    fig2.update_layout(title=dict(text="Main characteristics for genere", font=dict(size=40), automargin=True, yref='paper'))
    st.write(fig2)

if curtain1=='Explicit content':

    fig=plt.figure(figsize=(12,8))
    color=sb.color_palette('pastel6')
    explode=[0.2,0.1,0.1]
    plt.pie(dist_explicit,labels=dist_explicit.index,autopct='%.2f%%', startangle=90,
            wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' },colors=color,explode=explode)
    plt.title('Percentige of songs with explicit content',size=22)
    plt.legend()

    #plt.show()
    st.write(fig)


    fig0=plt.figure(figsize=(12,10))
    fig0 = px.pie(explicit_df, values="explicit", names="maingenere",hole=0.5,width =800,height=700)
    fig0.update_layout(title_text="How many songs with explicit content for genre")
    fig0.update_traces(textinfo='value+label+percent')
    #fig0.show()
    st.write(fig0)


if curtain1=='BPM':
    fig=plt.figure(figsize=(14,8))
    plt.plot(top_12_genere_gb.index,top_12_genere_gb.tempo,marker='o',markersize='12',linewidth=5)
    plt.xticks(rotation=45)
    plt.title('Mean BPM for genere',size=22)
    plt.xlabel('Maingenere')
    plt.ylabel('Mean BPM')
    st.write(fig)

#to add a sidebar
    
bb_main12_df=top_12_genere_df.copy()
songs_attribute_list=['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness','tempo']
bb_main12_df.loc[bb_main12_df.valence >= 0.6, 'valence'] = 1
bb_main12_df.loc[bb_main12_df.valence < 0.6, 'valence'] = 0

##like img

img_like=Image.open('like.png')
#st.write('thebug')
st.sidebar.image(img_like,width=50)

st.sidebar.write('Valence attribute - Vibe')
if st.sidebar.checkbox('Classifier'):
    st.title('Attribute identifying liked songs')

    for attribute in songs_attribute_list:

        like=bb_main12_df[bb_main12_df['valence']==1][attribute]
        dislike=bb_main12_df[bb_main12_df['valence']==0][attribute]

        fig=plt.figure(figsize=(12,7))

        sb.histplot(like,bins=30,label='Positive',color='Green',kde=True,stat='density')
        sb.histplot(dislike,bins=30,label='Negative',color='Red',kde=True,stat='density')

        plt.legend(loc='upper right')
        plt.title(f'Positive and negative histogram plot for {attribute} ')

        st.write(fig)

mask_like=bb_main12_df.valence==1
like_songs_df=bb_main12_df[mask_like]

like_songs_for_main_genere=like_songs_df.maingenere.value_counts()

mask_dislike=bb_main12_df.valence==0
dislike_songs_df=bb_main12_df[mask_dislike]

dislike_songs_for_maingenere=dislike_songs_df.maingenere.value_counts()

like_dislike_songs_df=pd.concat([dislike_songs_for_maingenere,like_songs_for_main_genere],axis=1)

like_dislike_songs_df.columns=['Dislike songs','Like songs']


if st.sidebar.checkbox('Like / Dislike songs for genre?'):
    fig4 = px.bar(like_dislike_songs_df, x=like_dislike_songs_df.index, y=['Dislike songs','Like songs'], barmode='group',text_auto=True)
    fig4.update_layout(title=dict(text="Like/Dislike songs for genere", font=dict(size=50), automargin=True, yref='paper'))
    st.write(fig4)

####
performer_songs_genere_df=pd.DataFrame(columns=['Performer','Song','maingenere','spotify_track_popularity'])
performer_songs_genere_df['Performer']=billboard_df.Performer
performer_songs_genere_df['Song']=billboard_df.Song
performer_songs_genere_df['maingenere']=billboard_df.maingenere
performer_songs_genere_df['spotify_track_popularity']=billboard_df.spotify_track_popularity

lista_main_generi_ridotta=['pop', 'rock', 'other', 'soul', 'rap', 'country', 'disco', 'jazz', 'blues', 'funk', 'dance', 'metal']
top_performer_song_per_populatiry_genere_df=pd.DataFrame()
for generi in lista_main_generi_ridotta:
    top_performer_song_per_populatiry_genere_series=performer_songs_genere_df[performer_songs_genere_df.maingenere==generi].sort_values(by='spotify_track_popularity',ascending=False).iloc[0]
    top_performer_song_per_populatiry_genere_df=pd.concat([top_performer_song_per_populatiry_genere_df,top_performer_song_per_populatiry_genere_series],ignore_index=True, axis = 1)
    
top_songs_for_genre_df=top_performer_song_per_populatiry_genere_df.T

mask_pop=top_songs_for_genre_df.maingenere=='pop'
mask_rock=top_songs_for_genre_df.maingenere=='rock'
mask_other=top_songs_for_genre_df.maingenere=='other'
mask_soul=top_songs_for_genre_df.maingenere=='soul'
mask_rap=top_songs_for_genre_df.maingenere=='rap'
mask_country=top_songs_for_genre_df.maingenere=='country'
mask_disco=top_songs_for_genre_df.maingenere=='disco'
mask_jazz=top_songs_for_genre_df.maingenere=='jazz'
mask_blues=top_songs_for_genre_df.maingenere=='blues'
mask_funk=top_songs_for_genre_df.maingenere=='funk'
mask_dance=top_songs_for_genre_df.maingenere=='dance'
mask_metal=top_songs_for_genre_df.maingenere=='metal'


#### grafico 
st.title('What about Songs?')

if st.checkbox('Best songs for populaity/genre'):
    fig=plt.figure(figsize=(26,6))
    col1=sb.color_palette('pastel6')
    col2=sb.color_palette('colorblind')
    col3=sb.color_palette('muted')



    plt.bar(top_songs_for_genre_df[mask_pop].Song,top_songs_for_genre_df[mask_pop].spotify_track_popularity,color=col1[1],width=0.4)
    plt.text(0,100,'Pop',ha='center',va='center',size='12')
    plt.bar(top_songs_for_genre_df[mask_rock].Song,top_songs_for_genre_df[mask_rock].spotify_track_popularity,color=col1[2],width=0.4)
    plt.text(1,100,'rock',ha='center',va='center',size='12')
    plt.bar(top_songs_for_genre_df[mask_other].Song,top_songs_for_genre_df[mask_other].spotify_track_popularity,color=col1[3],width=0.4)
    plt.text(2,100,'other',ha='center',va='center',size='12')
    plt.bar(top_songs_for_genre_df[mask_soul].Song,top_songs_for_genre_df[mask_soul].spotify_track_popularity,color=col1[4],width=0.4)
    plt.text(3,100,'soul',ha='center',va='center',size='12')
    plt.bar(top_songs_for_genre_df[mask_rap].Song,top_songs_for_genre_df[mask_rap].spotify_track_popularity,color=col2[1],width=0.4)
    plt.text(4,100,'rap',ha='center',va='center',size='12')
    plt.bar(top_songs_for_genre_df[mask_country].Song,top_songs_for_genre_df[mask_country].spotify_track_popularity,color=col2[2],width=0.4)
    plt.text(5,100,'country',ha='center',va='center',size='12')
    plt.bar(top_songs_for_genre_df[mask_disco].Song,top_songs_for_genre_df[mask_disco].spotify_track_popularity,color=col2[3],width=0.4)
    plt.text(6,100,'disco',ha='center',va='center',size='12')
    plt.bar(top_songs_for_genre_df[mask_jazz].Song,top_songs_for_genre_df[mask_jazz].spotify_track_popularity,color=col2[4],width=0.4)
    plt.text(7,100,'jazz',ha='center',va='center',size='12')
    plt.bar(top_songs_for_genre_df[mask_blues].Song,top_songs_for_genre_df[mask_blues].spotify_track_popularity,color=col3[1],width=0.4)
    plt.text(8,100,'blues',ha='center',va='center',size='12')
    plt.bar(top_songs_for_genre_df[mask_funk].Song,top_songs_for_genre_df[mask_funk].spotify_track_popularity,color=col3[2],width=0.4)
    plt.text(9,100,'funk',ha='center',va='center',size='12')
    plt.bar(top_songs_for_genre_df[mask_dance].Song,top_songs_for_genre_df[mask_dance].spotify_track_popularity,color=col3[3],width=0.4)
    plt.text(10,100,'dance',ha='center',va='center',size='12')
    plt.bar(top_songs_for_genre_df[mask_metal].Song,top_songs_for_genre_df[mask_metal].spotify_track_popularity,color=col3[4],width=0.4)
    plt.text(11,100,'metal',ha='center',va='center',size='12')

    plt.title("Popularity Vs Track Name",size=20)
    plt.xlabel('Top Songs for genere',size=18)
    plt.ylabel('Popularity',size=18)
    plt.xticks(rotation=90)
    plt.legend()
    st.write(fig)

##
    
performer_w_most_songs=top_12_genere_df.Performer.value_counts()
performer_w_most_songs_df = performer_w_most_songs.to_frame(name="number of songs")
performer_w_most_songs_df.reset_index(inplace=True)


if st.checkbox('Performer with the most songs'):
    plt.figure(figsize=(12,10))
    fig0 = px.pie(performer_w_most_songs_df[1:11], values="number of songs", names="Performer",hole=0.5,width =800,height=700)
    fig0.update_layout(title_text="10 Performer with most songs in the chart - Number of songs")
    fig0.update_traces(textinfo='value+label')
    st.write(fig0)

#
    
top_50_popular_song=billboard_df.sort_values(by='spotify_track_popularity',ascending=False).head(50)
top_50_popular_song_df=pd.DataFrame(columns=['Performer','Song','spotify_track_popularity','maingenere'])
top_50_popular_song_df['Performer']=top_50_popular_song.Performer
top_50_popular_song_df['Song']=top_50_popular_song.Song
top_50_popular_song_df['spotify_track_popularity']=top_50_popular_song.spotify_track_popularity
top_50_popular_song_df['maingenere']=top_50_popular_song.maingenere
top_50_popular_song_df['Performer - Song']=top_50_popular_song['Performer']+' - '+ top_50_popular_song['Song']

##

top_10_loudness_trak=billboard_df[['loudness','Song','Performer','maingenere']].sort_values(by='loudness',ascending=True)[:10]
top_10_loudness_trak['Performer-Song']=top_10_loudness_trak['Song']+ ' - ' +top_10_loudness_trak['Performer']

###

top_10_danceability_trak=billboard_df[['danceability','Song','Performer','maingenere']].sort_values(by='danceability',ascending=False)[:10]
top_10_danceability_trak['Performer-Song']=top_10_danceability_trak['Song']+ ' - ' +top_10_danceability_trak['Performer']
top_10_danceability_trak['Performer-Song'] = top_10_danceability_trak['Performer-Song'].str.replace('$','S')

####

top_10_energy_trak=billboard_df[['energy','Song','Performer','maingenere']].sort_values(by='energy',ascending=False)[:10]
top_10_energy_trak['Performer-Song']=top_10_energy_trak['Song']+ ' - ' +top_10_energy_trak['Performer']

#####

top_10_speechiness_trak=billboard_df[['speechiness','Song','Performer','maingenere']].sort_values(by='speechiness',ascending=False)[:10]
top_10_speechiness_trak['Performer-Song']=top_10_speechiness_trak['Song']+ ' - ' +top_10_speechiness_trak['Performer']

######

top_10_acousticness_trak=billboard_df[['acousticness','Song','Performer','maingenere']].sort_values(by='acousticness',ascending=False)[:10]
top_10_acousticness_trak['Performer-Song']=top_10_acousticness_trak['Song']+ ' - ' +top_10_acousticness_trak['Performer']

######

top_10_instrumentalness_trak=billboard_df[['instrumentalness','Song','Performer','maingenere']].sort_values(by='instrumentalness',ascending=False)[:10]
top_10_instrumentalness_trak['Performer-Song']=top_10_instrumentalness_trak['Song']+ ' - ' +top_10_instrumentalness_trak['Performer']

######

top_10_liveness_trak=billboard_df[['liveness','Song','Performer','maingenere']].sort_values(by='liveness',ascending=False)[:10]
top_10_liveness_trak['Performer-Song']=top_10_liveness_trak['Song']+ ' - ' +top_10_liveness_trak['Performer']

###
top_10_longest_trak=top_12_genere_df[['spotify_track_duration_minute','Song','Performer','maingenere']].sort_values(by='spotify_track_duration_minute',ascending=False)[:10]
top_10_longest_trak['Performer-Song']=top_10_longest_trak['Song']+ ' - ' +top_10_longest_trak['Performer']

###

top_10_liveness_trak=billboard_df[['liveness','Song','Performer','maingenere']].sort_values(by='liveness',ascending=False)[:10]
top_10_liveness_trak['Performer-Song']=top_10_liveness_trak['Song']+ ' - ' +top_10_liveness_trak['Performer']




curtain=st.selectbox('Click to see - Top 10 Songs main characteristics',('None','Top 10 Popular song - for genre','Top 10 Loudness songs - for genre',
                                     'Top 10 Danceability songs - for genre','Top 10 Energy songs - for genre',
                                     'Top 10 Speechiness songs - for genre','Top 10 Acousticness songs - for genre',
                                     'Top 10 Instrumentalness songs - for genre','Top 10 Liveness songs - for genre',
                                     'Top 10 Longest songs - for genre'))
if curtain == 'None':
    pass
if curtain=='Top 10 Popular song - for genre':
    fig=plt.figure(figsize=(12,8))
    ax=sb.scatterplot(data = top_50_popular_song_df[0:10], x = 'Performer - Song', y = 'spotify_track_popularity',hue='maingenere',s=600)

    plt.xticks(rotation=90)

    plt.setp(ax.get_legend().get_texts(), fontsize='18')
    plt.setp(ax.get_legend().get_title(), fontsize='18')
    plt.title('Top 10 Songs - Most popular',size=22)
    st.write(fig)


if curtain=='Top 10 Loudness songs - for genre':
    fig=plt.figure(figsize=(12,8))
    ax=sb.scatterplot(data = top_10_loudness_trak, x = 'Performer-Song', y = 'loudness',hue='maingenere',s=600)

    plt.xticks(rotation=90)

    plt.setp(ax.get_legend().get_texts(), fontsize='18')
    plt.setp(ax.get_legend().get_title(), fontsize='18')
    plt.title('Top 10 Songs - Most loud',size=22)
    st.write(fig)

if curtain=='Top 10 Danceability songs - for genre':
    fig=plt.figure(figsize=(12,8))
    ax=sb.scatterplot(data = top_10_danceability_trak, x = 'Performer-Song', y = 'danceability',hue='maingenere',s=600)

    plt.xticks(rotation=90)

    plt.setp(ax.get_legend().get_texts(), fontsize='18')
    plt.setp(ax.get_legend().get_title(), fontsize='18')
    plt.title('Top 10 Songs - Most danceability',size=22)
    st.write(fig)

if curtain=='Top 10 Energy songs - for genre':
    fig=plt.figure(figsize=(12,8))
    ax=sb.scatterplot(data = top_10_energy_trak, x = 'Performer-Song', y = 'energy',hue='maingenere',s=600)

    plt.xticks(rotation=90)

    plt.setp(ax.get_legend().get_texts(), fontsize='18')
    plt.setp(ax.get_legend().get_title(), fontsize='18')
    plt.title('Top 10 Songs - Most energy',size=22)
    st.write(fig)

if curtain=='Top 10 Speechiness songs - for genre':
    fig=plt.figure(figsize=(12,8))
    ax=sb.scatterplot(data = top_10_speechiness_trak, x = 'Performer-Song', y = 'speechiness',hue='maingenere',s=600)

    plt.xticks(rotation=90)

    plt.setp(ax.get_legend().get_texts(), fontsize='18')
    plt.setp(ax.get_legend().get_title(), fontsize='18')
    plt.title('Top 10 Songs - Most speechiness',size=22)
    st.write(fig)

if curtain=='Top 10 Acousticness songs - for genre':

    fig=plt.figure(figsize=(12,8))
    ax=sb.scatterplot(data = top_10_acousticness_trak, x = 'Performer-Song', y = 'acousticness',hue='maingenere',s=600)

    plt.xticks(rotation=90)

    plt.setp(ax.get_legend().get_texts(), fontsize='18')
    plt.setp(ax.get_legend().get_title(), fontsize='18')
    plt.title('Top 10 Songs - Most acousticness',size=22)

    st.write(fig)

if curtain=='Top 10 Instrumentalness songs - for genre':

    fig=plt.figure(figsize=(12,8))
    ax=sb.scatterplot(data = top_10_instrumentalness_trak, x = 'Performer-Song', y = 'instrumentalness',hue='maingenere',s=600)

    plt.xticks(rotation=90)

    plt.setp(ax.get_legend().get_texts(), fontsize='18')
    plt.setp(ax.get_legend().get_title(), fontsize='18')
    plt.title('Top 10 Songs - Most instrumentalness',size=22)

    st.write(fig)

if curtain=='Top 10 Instrumentalness songs - for genre':

    fig=plt.figure(figsize=(12,8))
    ax=sb.scatterplot(data = top_10_instrumentalness_trak, x = 'Performer-Song', y = 'instrumentalness',hue='maingenere',s=600)

    plt.xticks(rotation=90)

    plt.setp(ax.get_legend().get_texts(), fontsize='18')
    plt.setp(ax.get_legend().get_title(), fontsize='18')
    plt.title('Top 10 Songs - Most instrumentalness',size=22)

    st.write(fig)


if curtain=='Top 10 Liveness songs - for genre':

    fig=plt.figure(figsize=(12,8))
    ax=sb.scatterplot(data = top_10_liveness_trak, x = 'Performer-Song', y = 'liveness',hue='maingenere',s=600)

    plt.xticks(rotation=90)

    plt.setp(ax.get_legend().get_texts(), fontsize='18')
    plt.setp(ax.get_legend().get_title(), fontsize='18')
    plt.title('Top 10 Songs - Most liveness',size=22)

    st.write(fig)

if curtain=='Top 10 Longest songs - for genre':

    fig=plt.figure(figsize=(12,8))
    ax=sb.scatterplot(data = top_10_longest_trak, x = 'Performer-Song', y = 'spotify_track_duration_minute',hue='maingenere',s=600)

    plt.xticks(rotation=90)

    plt.setp(ax.get_legend().get_texts(), fontsize='18')
    plt.setp(ax.get_legend().get_title(), fontsize='18')
    plt.title('Top 10 Songs - Most Long song',size=22)

    st.write(fig)

###
    
mask_toll=top_12_genere_df.Performer=='Tool'
tool_df=top_12_genere_df[mask_toll]

mask_pf=top_12_genere_df.Performer=='Pink Floyd'
pf_df=top_12_genere_df[mask_pf]

mask_kl=top_12_genere_df.Performer=='Kendrick Lamar'
kl_df=top_12_genere_df[mask_kl]

mask_gorillaz=top_12_genere_df.Performer=='Gorillaz'
gorillaz_df=top_12_genere_df[mask_gorillaz]


##
img_nota=Image.open('nota2.png')

st.sidebar.image(img_nota,width=50)

st.sidebar.write('My library')

if st.sidebar.checkbox('Metal'):
    st.title('Tool')

    img_tool=Image.open('tool.png')

    st.image(img_tool,width=300)

    curtain=st.selectbox('Click to see - Tool songs characteristics',('None','Main characteristics of Tool Songs - attribute 1',
                                         'Main characteristics of Tool Songs - attribute 2',
                                         'Main characteristics of Tool Songs - attribute 3'))
    if curtain == 'None':
        pass
    if curtain=='Main characteristics of Tool Songs - attribute 1':
        fig1 = px.bar(tool_df, x=tool_df.Song, y=['danceability','energy','valence','speechiness','instrumentalness','liveness'], barmode='group',text_auto=True)
        fig1.update_layout(title=dict(text="Main characteristics of Tool Songs", font=dict(size=40), automargin=True, yref='paper'))
        st.write(fig1)

    if curtain=='Main characteristics of Tool Songs - attribute 2':
        fig1 = px.bar(tool_df, x=tool_df.Song, y=['tempo','spotify_track_popularity'], barmode='group',text_auto=True)
        fig1.update_layout(title=dict(text="Main characteristics of Tool Songs", font=dict(size=40), automargin=True, yref='paper'))
        st.write(fig1)

    if curtain=='Main characteristics of Tool Songs - attribute 3':
        fig1 = px.bar(tool_df, x=tool_df.Song, y=['spotify_track_duration_minute','loudness','key'], barmode='group',text_auto=True)
        fig1.update_layout(title=dict(text="Main characteristics of Tool Songs", font=dict(size=40), automargin=True, yref='paper'))
        st.write(fig1)


if st.sidebar.checkbox('Rock'):
    st.title('Pink Floyd')

    img_pf=Image.open('pf.png')

    st.image(img_pf,width=300)

    curtain=st.selectbox('Click to see - Pink Floyd songs characteristics',('None','Main characteristics of Pink Floyd Songs - attribute 1',
                                         'Main characteristics of Pink Floyd Songs - attribute 2',
                                         'Main characteristics of Pink Floyd Songs - attribute 3'))
    if curtain == 'None':
        pass
    
    if curtain=='Main characteristics of Pink Floyd Songs - attribute 1':
        fig1 = px.bar(pf_df, x=pf_df.Song, y=['danceability','energy','valence','speechiness','instrumentalness','liveness'], barmode='group',text_auto=True)
        fig1.update_layout(title=dict(text="Main characteristics of Pinck Floyd Songs", font=dict(size=30), automargin=True, yref='paper'))
        st.write(fig1)
    
    if curtain=='Main characteristics of Pink Floyd Songs - attribute 2':
        fig1 = px.bar(pf_df, x=pf_df.Song, y=['tempo','spotify_track_popularity'], barmode='group',text_auto=True)
        fig1.update_layout(title=dict(text="Main characteristics of Pink Floyd Songs", font=dict(size=30), automargin=True, yref='paper'))
        st.write(fig1)

    if curtain=='Main characteristics of Pink Floyd Songs - attribute 3':
        fig1 = px.bar(pf_df, x=pf_df.Song, y=['spotify_track_duration_minute','loudness','key'], barmode='group',text_auto=True)
        fig1.update_layout(title=dict(text="Main characteristics of Tool Songs", font=dict(size=30), automargin=True, yref='paper'))
        st.write(fig1)


if st.sidebar.checkbox('Rap'):
    st.title('Kendrick Lamar')

    img_kl=Image.open('kl.png')

    st.image(img_kl,width=300)

    curtain=st.selectbox('Click to see - Kendrick Lamar songs characteristics',('None','Main characteristics of Kendrick Lamar Songs - attribute 1',
                                         'Main characteristics of Kendrick Lamar Songs - attribute 2',
                                         'Main characteristics of Kendrick Lamar Songs - attribute 3'))
    if curtain1 == 'None':
        pass
    
    if curtain=='Main characteristics of Kendrick Lamar Songs - attribute 1':
        fig1 = px.bar(kl_df, x=kl_df.Song, y=['danceability','energy','valence','speechiness','instrumentalness','liveness'], barmode='group')
        fig1.update_layout(title=dict(text="Main characteristics of Kendrick Lamar Songs", font=dict(size=30), automargin=True, yref='paper'))
        st.write(fig1)

    if curtain=='Main characteristics of Kendrick Lamar Songs - attribute 2':
        fig1 = px.bar(kl_df, x=kl_df.Song, y=['tempo','spotify_track_popularity'], barmode='group',text_auto=True)
        fig1.update_layout(title=dict(text="Main characteristics of Kendrick Lamar Songs", font=dict(size=30), automargin=True, yref='paper'))
        st.write(fig1)

    if curtain=='Main characteristics of Kendrick Lamar Songs - attribute 3':
        fig1 = px.bar(kl_df, x=kl_df.Song, y=['spotify_track_duration_minute','loudness','key'], barmode='group',text_auto=True)
        fig1.update_layout(title=dict(text="Main characteristics of Kendrick Lamar Songs", font=dict(size=30), automargin=True, yref='paper'))
        st.write(fig1)
    

if st.sidebar.checkbox('Pop'):
    st.title('Gorillaz')

    img_gor=Image.open('gorillaz.png')

    st.image(img_gor,width=300)

    curtain=st.selectbox('Click to see - Gorillaz songs characteristics',('None','Main characteristics of Gorillaz Songs - attribute 1',
                                         'Main characteristics of Gorillaz Songs - attribute 2',
                                         'Main characteristics of Gorillaz Songs - attribute 3'))
    if curtain1 == 'None':
        pass
    
    if curtain=='Main characteristics of Gorillaz Songs - attribute 1':
        fig1 = px.bar(gorillaz_df, x=gorillaz_df.Song, y=['danceability','energy','valence','speechiness','instrumentalness','liveness'], barmode='group',text_auto=True)
        fig1.update_layout(title=dict(text="Main characteristics of Gorillaz Songs", font=dict(size=30), automargin=True, yref='paper'))
        st.write(fig1)

    if curtain=='Main characteristics of Gorillaz Songs - attribute 2':
        fig1 = px.bar(gorillaz_df, x=gorillaz_df.Song, y=['tempo','spotify_track_popularity'], barmode='group',text_auto=True)
        fig1.update_layout(title=dict(text="Main characteristics of Gorillaz Songs", font=dict(size=30), automargin=True, yref='paper'))
        st.write(fig1)
    if curtain=='Main characteristics of Gorillaz Songs - attribute 3':
        fig1 = px.bar(gorillaz_df, x=gorillaz_df.Song, y=['spotify_track_duration_minute','loudness','key'], barmode='group',text_auto=True)
        fig1.update_layout(title=dict(text="Main characteristics of Gorillaz Songs", font=dict(size=30), automargin=True, yref='paper'))
        st.write(fig1)
    

## CLUSTER
        
column_to_use=['danceability','energy','key','loudness','mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness','valence','time_signature','tempo','spotify_track_duration_minute']

cluster_df=top_12_genere_df[column_to_use]


ssd=[]

for k in range(1,13):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(cluster_df)
    ssd.append(kmeans.inertia_)


st.title('Cluster Analysis')

# 

st.markdown("""

    We are interested in group similar observation, where similar observation will go in the same group.
    So we want **homogeneity within the group** and **heterogeneity between groups**
            
    First of all we consider only the numeric variable
            
            """)

st.write(cluster_df.head(3))

cluster_df=(cluster_df-cluster_df.mean()/cluster_df.std())

st.markdown("""

#### **How many cluster we have to consider?**

            """)


fig=plt.figure(figsize=(10,6))
plt.plot(range(1,13),ssd)
plt.title('Elbow Method for Clustering')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.show()
st.write(fig)

st.markdown("""
#### **We can see that our 'Elbow point' is at k=4, so we try to build a model using 4 cluster**            
            """)

kmeans=KMeans(n_clusters=4,random_state=1)
kmeans.fit(cluster_df)

clusters=kmeans.predict(cluster_df)

pca=PCA(n_components=2)
df_2d=pca.fit_transform(cluster_df)

fig=plt.figure(figsize=(21,12))
plt.scatter(df_2d[:, 0], df_2d[:, 1], c=clusters)
plt.title('Clustering Of Spotify Songs')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
st.write(fig)

st.markdown("""
#### How to evalute if the model that we have built is a good model ?
            
    Silhouette method: is an index that move from -1 to 1,provides a measure of 
    how similar an object is to the cluster it belongs to compared to other clusters

            """)

st.write(silhouette_score(cluster_df,clusters))

st.markdown("""
In general 0.49 is not a bad result but as we can see, the characteristics of the clusters are very similar
            """)

four_cluster_df=cluster_df.copy()
four_cluster_df['cluster']=clusters
if st.checkbox('Numeric cluster characteristics'):
    cluster_mean=four_cluster_df.groupby('cluster').mean()
    st.write(cluster_mean)

###

etichette=kmeans.labels_

cluster1=top_12_genere_df[etichette==0]
c1_n_songs_maingenere=cluster1.maingenere.value_counts()
cluster2=top_12_genere_df[etichette==1]
c2_n_songs_maingenere=cluster2.maingenere.value_counts()
cluster3=top_12_genere_df[etichette==2]
c3_n_songs_maingenere=cluster3.maingenere.value_counts()
cluster4=top_12_genere_df[etichette==3]
c4_n_songs_maingenere=cluster4.maingenere.value_counts()

c1234_df=pd.concat([c1_n_songs_maingenere,c2_n_songs_maingenere,c3_n_songs_maingenere,c4_n_songs_maingenere],axis=1)
c1234_df.columns=['c1','c2','c3','c4']

####

four_cluster_df=cluster_df.copy()
four_cluster_df['cluster']=clusters

cluster_mean=four_cluster_df.groupby('cluster').mean()

cluster_mean_wo_tempo=cluster_mean.drop('tempo', axis=1)


curtain=st.selectbox('Click to see - Cluster characteristics',('None','Cluster distribution','Cluster characteristics1','Cluster characteristics2'))

if curtain == 'None':
    pass

if curtain=='Cluster distribution':

    
   
    unique_clusters = np.unique(clusters)


    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14,14), sharex=True, sharey=True)


    axs = axs.flatten()


    for i, cluster in enumerate(unique_clusters):
    
        df_cluster = df_2d[clusters == cluster]

    
        df_other_clusters = df_2d[clusters != cluster]

        
        axs[i].scatter(df_other_clusters[:, 0], df_other_clusters[:, 1], c='gray', label='Other clusters', alpha=0.5)

        
        axs[i].scatter(df_cluster[:, 0], df_cluster[:, 1], c='red', label='Cluster {}'.format(cluster))

    
        axs[i].set_xlabel('Component 1')
        axs[i].set_ylabel('Component 2')

        axs[i].legend()

    st.write(fig)


if curtain=='Cluster characteristics1':

    fig=plt.figure(figsize=(12,8))
    cluster_mean_wo_tempo.plot(kind='bar')
    plt.title('Cluster Characteristics')
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xticks(rotation=60)
    #t.pyplot(fig)
    st.pyplot(fig=plt)

if curtain=='Cluster characteristics2':

    fig=plt.figure(figsize=(12,8))
    cluster_mean['tempo'].plot(kind='bar')
    plt.title('Cluster Characteristics')
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xticks(rotation=60)
    st.pyplot(fig=plt)

##
    

    
curtain=st.selectbox('Click to see - What we find in clusters',('None','Distribution of genre for cluster 1','Distribution of genre for cluster 2',
                                     'Distribution of genre for cluster 3','Distribution of genre for cluster 4',
                                     'Maingenre distribution for cluster'))

if curtain == 'None':
    pass
    
if curtain=='Distribution of genre for cluster 1':
    fig0=plt.figure(figsize=(12,8))
    fig0 = px.pie(c1234_df, values="c1", names=c1234_df.index,hole=0.5,width =800,height=700)
    fig0.update_layout(title_text="Distribution of genre for cluster 1")
    fig0.update_traces(textinfo='value+label+percent')
    #fig0.show()
    st.write(fig0)

if curtain=='Distribution of genre for cluster 2':
    fig0=plt.figure(figsize=(12,8))
    fig0 = px.pie(c1234_df, values="c2", names=c1234_df.index,hole=0.5,width =800,height=700)
    fig0.update_layout(title_text="Distribution of genre for cluster 2")
    fig0.update_traces(textinfo='value+label+percent')
    #fig0.show()
    st.write(fig0)

if curtain=='Distribution of genre for cluster 3':
    fig0=plt.figure(figsize=(12,8))
    fig0 = px.pie(c1234_df, values="c2", names=c1234_df.index,hole=0.5,width =800,height=700)
    fig0.update_layout(title_text="Distribution of genre for cluster 3")
    fig0.update_traces(textinfo='value+label+percent')
    #fig0.show()
    st.write(fig0)

if curtain=='Distribution of genre for cluster 4':
    fig0=plt.figure(figsize=(12,8))
    fig0 = px.pie(c1234_df, values="c2", names=c1234_df.index,hole=0.5,width =800,height=700)
    fig0.update_layout(title_text="Distribution of genre for cluster 4")
    fig0.update_traces(textinfo='value+label+percent')
    #fig0.show()
    st.write(fig0)



if curtain=='Maingenre distribution for cluster':
    fig1 = px.bar(c1234_df, x=c1234_df.index, y=['c1','c2','c3','c4'], barmode='group',text_auto=True)

    fig1.update_layout(title=dict(text="Distribution of genre for each cluster", font=dict(size=30), automargin=True, yref='paper'))

    st.write(fig1)





    









        









        




