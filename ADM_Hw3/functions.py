import nltk
import json
import math
import operator
import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer
from scipy.spatial.distance import cosine
from scipy.spatial import distance
from nltk.corpus import stopwords
from IPython.display import HTML
from nltk.stem.porter import *



def create_tsv(): 
    f = open('Airbnb_Texas_Rentals.csv', 'r', encoding='utf-8')
    i = 0
    for line in f: 
        with open('doc_' + str(i) + '.tsv', 'w+', encoding = 'utf-8') as ftmp: 
            ftmp.write(re.sub(r',','\t', line))
        i = i +1

def clean(t): 
    porter = PorterStemmer()
    sentence = t.lower()
    sentence=sentence.replace('\n','')
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    wn1=WordNetLemmatizer()
    r=[ wn1.lemmatize(w) if wn1.lemmatize(w).endswith('e') else porter.stem(w)  

for w in filtered_words]      
    return r


def tag(list):     
    words = nltk.word_tokenize(list)
    tagged = nltk.pos_tag(words)             
    a = [word for word,tag in tagged if tag not in ('VB')]
    return a


def create_vocabulary(l):
    vocabulary = {}
    vocabulary = {l[i]: i for i in range(len(l))}    
    #save vocabulary
    with open('vocabulary.txt', 'w') as fp:
        json.dump(vocabulary, fp)

def create_index(l, index):     
    #load vocaulary 
    with open('vocabulary.txt') as fp:
        vocabulary = json.load(fp)     
    for word in l: 
        if vocabulary[word] in term_index:
            term_index[vocabulary[word]].append(index)
        else:
             term_index[vocabulary[word]] = [index]

def find_result_with_index(cleaned): 
    #load vocabulary    
    with open('vocabulary.txt') as fp:
        voc = json.load(fp)            
    #load term_index
    with open('term_index.txt') as fp:
        index = json.load(fp)
    result = []    
    for i in cleaned:  
        try: 
            result.append(index[str(voc[i])])       
        except: 
            continue    
    return result

def load_result(result, outputIndex):     
    temp_df = pd.DataFrame(index = None)      
    end = outputIndex + 15
    file = pd.read_csv('Airbnb_Texas_Rentals2.csv')
    for i in result: 
        for j in range(outputIndex, end): 
            try: 
                a = file.loc[(file['Unnamed: 0']).astype(str) == str(i[j])]
            except: 
                continue
            temp_df = temp_df.append(a[['title', 'description', 'city', 

'url']], sort = False,ignore_index=True) 
    temp_df = temp_df.drop_duplicates(subset='title', keep='first', inplace=False)       
    temp_df = temp_df.rename(columns={'title': 'Title', 'description':'Description', 

'city':'City', 'url':'URL'})
    return(temp_df)


def heap_sort(a,b,x):   

    #left child = 2*i + 1
    left = 2*x + 1
    #right child = 2*i + 2
    right = 2*x + 2  

    #assign current Index as the largest number
    l = x
    try: 
        if(a[l] < a[left]): 
            #largest index is now left
            l = left
    except: 
        pass
    try: 
        if(a[l] < a[right]): 
            #largest index is now right
            l = right
    except: 
        pass
    #If the largest value has changed from the current Index
    if(l != x):
        #swap 
        a[l], a[x] = a[x], a[l]
        b[l], b[x] = b[x], b[l]
        #do heap sort again
        heap_sort(a,b,l)

def load_result_newScore(result):      
    temp_df = pd.DataFrame(index = None)   

    file = pd.read_csv('Airbnb_Texas_Rentals2.csv')

    for i in range(len(result)):         
        a = file.loc[(file['Unnamed: 0']).astype(str) == str(result[i])]        
        a.insert(10, 'Ranking', (i+1) , allow_duplicates=True)    
        temp_df = temp_df.append(a[['Ranking','title', 'description', 'city', 

'url']], sort = False, ignore_index = True)                

    temp_df = temp_df.rename(index=None, columns={'title': 'Title', 

'description':'Description', 'city':'City', 'url':'URL'})        
    return(temp_df)




def d(doc_number,lis):
    su=0
    for j in lis:
        for m in j:
            if m[0]==doc_number:
                su+=m[1]**2
    return math.sqrt(su)

def list_query(l):
    r=[]
    while l!=[]:
        i=l[0]
        n=l.count(i)
        r.append(n)
        l.remove(i)
    return r

def norma_q(l):
    s=0
    for i in l:
        s+=i**2
    return math.sqrt(s)

def cosine_similary(q,nq,doc,query):
    #nq=norma_q(q)
    with open('d_norma.txt') as fp:
        d_norma = json.load(fp) 
    with open('inverted_index2.txt') as fp:
        inverted_index2=json.load(fp) 
    nd=d_norma[str(doc)]
    s=0
    e=0
    for i in query:
        l=inverted_index2[str(i)]
        for j in l:
            if j[0]==doc:
                s+=q[e]*j[1]
                e+=1
    res=1-s/(nq*nd)
    return res

def load_result_inverted_index(result, outputIndex):     
    temp_df = pd.DataFrame(index = None)   

    end = outputIndex + 15
    file = pd.read_csv('Airbnb_Texas_Rentals2.csv', delimiter = ',',  encoding 

= "utf-8")

    pattern = r'\\[a-z]?'

    for i in range(outputIndex, end):         
        a = file.loc[(file['Unnamed: 0']).astype(str) == str(result[i][0])]  
        a.insert(10, 'Similarity', float(round(result[i][1],2)), 

allow_duplicates=True)    
        temp_df = temp_df.append(a[['title', 'description', 'city', 'url', 

'Similarity']], sort = False, ignore_index = True)        
  

    temp_df = temp_df.drop_duplicates(subset='title', keep='first', 

inplace=False)
    temp_df = temp_df.rename(index=str, columns={'title': 'Title', 

'description':'Description', 'city':'City', 'url':'URL'})        
    return(temp_df)
