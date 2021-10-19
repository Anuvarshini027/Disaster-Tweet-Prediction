
## All purpose library
import pandas as pd
import numpy as np

## NLP library
import re
import string
import nltk
from nltk.corpus import stopwords

## ML Library
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score


## Visualization library
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

## Ignoring Warning during trainings 
import warnings
warnings.filterwarnings('ignore')

import streamlit as st


# In[ ]:


st.title('DISASTER TWEET PREDICTION USING NLP')
st.subheader('Upload the Training and Testing Dataset: (.csv)')
# creating a side bar 
st.sidebar.info("Created By : Team 18")
# Adding an image to the side bar 
st.sidebar.subheader("Contact Information : ")
col1, mid, col2 = st.sidebar.columns([1,1,20])
with col1:
    st.sidebar.subheader("Github : ")
with col2:
    st.sidebar.markdown("[![Github](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJGtP-Pq0P67Ptyv3tB7Zn2ZYPIT-lPGI7AA&usqp=CAU)](https://github.com/Anuvarshini027)")


file1 = st.file_uploader('Training Dataset')
file2 = st.file_uploader('Testing Dataset')
if file1,file2 is not None:
    train = pd.read_csv(file1)
    test=pd.read_csv(file2)
    st.header('**DISASTER TWEETS PREDICTION**')
    test1=test.copy()
    st.write(train.head())
    st.write("Size of training data: ",train.shape)
    st.write(test.head())
    st.write("Size of testing data: ",test.shape)
    st.write("Train Dataset missing data:\n",train.isnull().sum())
    st.write("Test Dataset missing data:\n ",test.isnull().sum())
    VCtrain=train['target'].value_counts().to_frame()

    ## seaborn barplot to display barchart
    vctrain=train["target"].value_counts()
    st.subheader("Labels distribution")
    st.bar_chart(vctrain)
    st.write(vctrain.to_frame)
    
    ## Going deep into disaster Tweets
    st.subheader("Data Pre-Processing")
    st.write("Random sample of disaster tweets: ",train[train.target==1].text.sample(3).to_frame())
    st.write("Random sample of non disaster tweets:",train[train.target==0].text.sample(3).to_frame())
    common_keywords=train["keyword"].value_counts()[:20].to_frame()
    st.subheader("Common keywords")
    st.write(common_keywords)
    st.write(train.location.value_counts()[:10].to_frame())
    # lowering the text
    train.text=train.text.apply(lambda x:x.lower() )
    test.text=test.text.apply(lambda x:x.lower())

    #removing square brackets
    # Deletes particular pattern 
    train.text=train.text.apply(lambda x:re.sub('\[.*?\]', '', x) )
    test.text=test.text.apply(lambda x:re.sub('\[.*?\]', '', x) )
    train.text=train.text.apply(lambda x:re.sub('<.*?>+', '', x) )
    test.text=test.text.apply(lambda x:re.sub('<.*?>+', '', x) )

    #removing hyperlink
    train.text=train.text.apply(lambda x:re.sub('https?://\S+|www\.\S+', '', x) )
    test.text=test.text.apply(lambda x:re.sub('https?://\S+|www\.\S+', '', x) )

    #removing puncuation
    train.text=train.text.apply(lambda x:re.sub('[%s]' % re.escape(string.punctuation), '', x) )
    test.text=test.text.apply(lambda x:re.sub('[%s]' % re.escape(string.punctuation), '', x) )
    train.text=train.text.apply(lambda x:re.sub('\n' , '', x) )
    test.text=test.text.apply(lambda x:re.sub('\n', '', x) )

    #remove words containing numbers
    train.text=train.text.apply(lambda x:re.sub('\w*\d\w*' , '', x) )
    test.text=test.text.apply(lambda x:re.sub('\w*\d\w*', '', x) )
    
    #After Data Cleaning
    st.write(train.text.head())
    st.write(test.text.head())
    
    disaster_tweets = train[train['target']==1]['text']
    non_disaster_tweets = train[train['target']==0]['text']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[16, 8])
    wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(disaster_tweets))
    _=ax1.imshow(wordcloud1)
    ax1.axis('off')
    ax1.set_title('Disaster Tweets',fontsize=40);

    wordcloud2 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(non_disaster_tweets))
    _=ax2.imshow(wordcloud2)
    ax2.axis('off')
    ax2.set_title('Non Disaster Tweets',fontsize=40);
    st.pyplot(fig)
    
    #Tokenizer
    token=nltk.tokenize.RegexpTokenizer(r'\w+')
    #applying token
    train.text=train.text.apply(lambda x:token.tokenize(x))
    test.text=test.text.apply(lambda x:token.tokenize(x))
    #view
    st.subheader("After tokenizing")
    st.write(train.text.head())
    st.write(test.text.head())
    
    nltk.download('stopwords')
    #removing stop words
    train.text=train.text.apply(lambda x:[w for w in x if w not in stopwords.words('english')])
    test.text=test.text.apply(lambda x:[w for w in x if w not in stopwords.words('english')])
    #view
    st.subheader("Stop Words in English")
    st.write(stopwords.words('english'))
    st.write("Total Number of Stopwords in English: ",len(stopwords.words('english')))
    
    st.subheader("After removing Stop Words in English")
    st.write(train.text.head())
    st.write(test.text.head())
    
    #stemmering the text and joining
    stemmer = nltk.stem.PorterStemmer()
    train.text=train.text.apply(lambda x:" ".join(stemmer.stem(token) for token in x))
    test.text=test.text.apply(lambda x:" ".join(stemmer.stem(token) for token in x))
    #View
    st.subheader("After Stemming")
    st.write(train.text.head())
    st.write(test.text.head())
    
    #one hot encoding
    count_vectorizer = CountVectorizer()
    train_vectors_count = count_vectorizer.fit_transform(train['text'])
    test_vectors_count = count_vectorizer.transform(test["text"])
        
    # Fitting a simple Naive Bayes
    NB_Vec = MultinomialNB()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    scores = cross_val_score(NB_Vec, train_vectors_count, train["target"], cv=cv, scoring="f1")
    #st.write("Scores: ",scores)
    NB_Vec.fit(train_vectors_count, train["target"])
    
    pred=NB_Vec.predict(test_vectors_count)
    pred_df=pd.DataFrame(pred,columns=['target'])   
    pred_df["Tweet"]=test1['text']
    
    a = st.slider('From ', min_value=0, max_value=pred_df.shape[0], value=1)
    b = st.slider('To ', min_value=a+1, max_value=pred_df.shape[0], value=15)
    st.subheader("After Prediction: ")
    st.write(pred_df.iloc[a:b,:])
    
    if(st.button("FINISH")):
        st.info("Thank You for your Patience!")
        st.balloons()

else:
    st.warning("No file has been chosen yet")

