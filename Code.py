#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import time
import re
import json
import os
import nltk


# In[2]:


def ecrire_json(chemin, contenu):
  w = open(chemin, "a", encoding="utf-8") #a et pas w pour pouvoir écrire par la suite sans remplacer l'ancien contenu
  w.write(json.dumps(contenu, indent=2, ensure_ascii=False))
  w.close()
#ensure_ascii = False (pour pouvoir afficher des caractères autres que ascii), pas forcément necessaire ici mais on ne perd rien à le mettre
#indent = 2 pour indenter (pas obligatoire mais rend plus lisible)


# In[3]:


#corpus 1
data_spam = pd.read_csv("Corpus/spamham.csv")
print(data_spam)
print(data_spam["text"]) #les textes, tout ce qu'on avait dans la première colonne
print(data_spam["spam"])
corpus1 = "data_spam"

#corpus 2
fr_tweets = pd.read_csv("Corpus/french_tweets.csv")
print(fr_tweets)
print(fr_tweets["text"])
print(fr_tweets["label"])
corpus2 = "fr_tweets"


# In[4]:


## récupérer X(instances) et y(classes) pour chaque corpus
X1 = data_spam["text"]
y1 = data_spam["spam"]
X2 = fr_tweets["text"]
y2 = fr_tweets["label"]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
V = CountVectorizer(ngram_range=(1, 2)) 
from nltk.corpus import stopwords #pour utiliser une liste de stopwords pour le français + modifier cellule en dessous
stopwordsFr = nltk.corpus.stopwords.words('french')
V2 = CountVectorizer(ngram_range=(1, 2), stop_words=stopwordsFr)


#pré-traitement
X1 = V.fit_transform(X1)
X2 = V.fit_transform(X2)
#X2 = V2.fit_transform(X2) si on veut utiliser, décommenter et commenter la ligne au dessus, et changer V par V2 pour le second corpus


## pour après pouvoir séparer train test
from sklearn.model_selection import train_test_split


# In[5]:


#classifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

liste_classifieurs= [
    ["Perceptron", Perceptron(eta0=0.1, random_state=0)], #random_state=0 pour que la séparation train-test ne se fasse pas au hasard (si hasard, différent à chaque fois et donc les résultats vont varier à chaque fois)
    ["Decision Tree", DecisionTreeClassifier()],          #eta0 multiplie les poids durant le pré-traitement avec fit par la valeur, ici 0,1
    ["SVM", SVC(gamma="scale")],
    ["KNN", KNeighborsClassifier()],
]

liste_classifieurs1= [
    ["Perceptron", Perceptron(eta0=0.1, random_state=0)],
    ["Decision Tree", DecisionTreeClassifier()],
    ["KNN", KNeighborsClassifier()],
]

#Pour l'évaluation
from sklearn.metrics import classification_report #pour precision, rappel, f-mesure, etc.
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

for nom, classifieur in liste_classifieurs:
    nom_json = "Resultats/spam/{}+{}.json".format(nom, V)
    if os.path.exists(nom_json) == True :
        print(nom_json, "déjà fait")
        continue
    liste_mots_index = V.get_feature_names() 
    start1 = time.perf_counter()
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=0) #on sépare train et test
    #X1_train, X1_test, y1_train, y1_test = train_test_split(X1_test, y1_test, test_size=0.3, random_state=0) #echantillon, à enlever lors des vrais tests
    classifieur.fit(X1_train, y1_train)
    y1_pred = classifieur.predict(X1_test)
    end1 = time.perf_counter()
    nom_classes = ["ham", "spam"]
    report = classification_report(y1_test, y1_pred, target_names=nom_classes, digits=4)
    ecrire_json(nom_json, report)
    d = "Durée : %f"%(end1-start1)
    ecrire_json(nom_json, d)
    nom_figure = "Figures/spam/{}+{}.png".format(nom, V)
    matrice_confusion = confusion_matrix(y1_test, y1_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    classes = ["ham", "spam"]
    sns.heatmap(matrice_confusion, cmap = plt.cm.Reds, 
            xticklabels=classes, yticklabels=classes, 
            annot=True, fmt ="d")
    plt.savefig(nom_figure)
    plt.show

for nom, classifieur in liste_classifieurs1:
    nom_json = "Resultats/tweets/{}+{}.json".format(nom, V)
    if os.path.exists(nom_json) == True :
        print(nom_json, "déjà fait")
        continue
    liste_mots_index = V.get_feature_names() 
    start2 = time.perf_counter()
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=0) #on sépare train et test
    #X2_train, X2_test, y2_train, y2_test = train_test_split(X2_test, y2_test, test_size=0.3, random_state=0)#echantillon, à enlever lors des vrais tests
    classifieur.fit(X2_train, y2_train)
    y2_pred = classifieur.predict(X2_test)
    end2 = time.perf_counter()
    nom_classes = ["negatif", "positif"]
    report = classification_report(y2_test, y2_pred, target_names=nom_classes, digits=4)
    ecrire_json(nom_json, report)
    h = "Durée : %f"%(end2-start2)
    ecrire_json(nom_json, h)
    nom_figure = "Figures/tweets/{}+{}.png".format(nom, V)
    matrice_confusion = confusion_matrix(y2_test, y2_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    classes = ["negatif", "positif"] #label: Polarity of the tweet (0 = negative, 1 = positive)
    sns.heatmap(matrice_confusion, cmap = plt.cm.Reds, 
            xticklabels=classes, yticklabels=classes, 
            annot=True, fmt ="d")
    plt.savefig(nom_figure)
    plt.show

# On fait la somme de tous les cas où la valeur dans y_test est bien trouvée dans y_pred
    print(nom, "appliqué sur", corpus1, "avec", V)
    print('Bons résultats: %d' % (y1_test == y1_pred).sum())
    print('Erreurs: %d' % (y1_test != y1_pred).sum())
    print("Durée : %f"%(end1-start1))
    print("\n")
    print(nom, "appliqué sur", corpus2, "avec", V)
    print('Bons résultats: %d' % (y2_test == y2_pred).sum())
    print('Erreurs: %d' % (y2_test != y2_pred).sum())
    print("Durée : %f"%(end2-start2))
    print("\n")


# In[ ]:




