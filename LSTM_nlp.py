# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 22:59:54 2018

@author: charl
"""

"""NETTOYAGE DU TEXTE"""
import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
import os
os.chdir("C:/Users/charl/OneDrive/Documents/jedha/full_time_exo/S7")

# on charge les documents en mémoire
def load_doc(filename):
	# on ouvre les fichiers 
	file = open(filename, mode='rt', encoding='utf-8')
	# on ouvre tous les textes
	text = file.read()
	# on ferme les fichiers
	file.close()
	return text

# On sépare un document en mémoire en phrases à chaque changement de ligne
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs

# Nettoyer une liste de lignes
def clean_pairs(lines):
	cleaned = list()
	# On prépare la focntion regex pour filtrer certains caractères
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# preparer la table de traduction pour retirer la ponctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# on normalise les caractères en ascii
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# on produit des tokens de mots en utilisant les espaces comme séparateurs
			line = line.split()
			# On passe en minuscule
			line = [word.lower() for word in line]
			# on enleve la ponctuation de chaque token
			line = [word.translate(table) for word in line]
			# on supprime tous les caractères non imprimables
			line = [re_print.sub('', w) for w in line]
			# On supprime les tokens contenant des chiffres
			line = [word for word in line if word.isalpha()]
			# enfin on enregistre les token comme des chaines de caractères
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)

# on sauvegarde un fichier de phrases nettoyées
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# chargement des données
filename = 'fra.txt'
doc = load_doc(filename)
# on rassemble les pairs anglais français
pairs = to_pairs(doc)
# nettoyage des paires
clean_pairs = clean_pairs(pairs)
# on enristre les paires nettoyées dans un fichier
save_clean_data(clean_pairs, 'english-french.pkl')
# vérification
for i in range(100):
	print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))


"""SEPARATION EN TRAIN TEST"""    
from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle

# chargement des données nettoyées
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# sauvegarde d'une liste de phrase nettoyées dans un fichier
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# chargement des données
raw_dataset = load_clean_sentences('english-french.pkl')

# reduction de la taille des données
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]
# on réordonne les données au hasard
shuffle(dataset)
# séparation en train / test
train, test = dataset[:9000], dataset[9000:]
# sauvegarde
save_clean_data(dataset, 'english-french-both.pkl')
save_clean_data(train, 'english-french-train.pkl')
save_clean_data(test, 'english-french-test.pkl')

"""APPRENTISSAGE DU MODELE"""
from pickle import load
from numpy import array
import pydot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

# chargement des données nettoyées
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# On transforme chaque mot en token
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# on définit une longueur de phrase maximale
def max_length(lines):
	return max(len(line.split()) for line in lines)

# on encode et on pad les sequences de mots pour qu'elles aient toutes le même format
def encode_sequences(tokenizer, length, lines):
	# On encode les séquences de mots comme des séquences de chiffre (pour que le modèle puisse les traiter plus facilement)
	X = tokenizer.texts_to_sequences(lines)
	# On remplit les espaces laissés vides par des 0 pour que toutes les phrases aient la longueur maximale
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# on encode la phrase cible 
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

# On defnit le modèle utilisant les LSTM
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

# chargement des données
dataset = load_clean_sentences('english-french-both.pkl')
train = load_clean_sentences('english-french-train.pkl')
test = load_clean_sentences('english-french-test.pkl')

# préparation du tokenizer anglais
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# préparation du tokenizer français
fra_tokenizer = create_tokenizer(dataset[:, 1])
fra_vocab_size = len(fra_tokenizer.word_index) + 1
fra_length = max_length(dataset[:, 1])
print('french Vocabulary Size: %d' % fra_vocab_size)
print('french Max Length: %d' % (fra_length))

# préparation des données d'apprentissage
trainX = encode_sequences(fra_tokenizer, fra_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# préparation des données de validation
testX = encode_sequences(fra_tokenizer, fra_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)

# Définition du modèle
model = define_model(fra_vocab_size, eng_vocab_size, fra_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# on print un résumé du modèle ainsi défini
print(model.summary())
import pydot
plot_model(model, to_file='model.png', show_shapes=True)
# Calcule du modèle
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)

"""EVALUATION DU MODELE"""
from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

# chargement des données nettoyées
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# longueur de phrase maximale
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encodage et padding des phrases
def encode_sequences(tokenizer, length, lines):
	# encodage des mots en chiffres
	X = tokenizer.texts_to_sequences(lines)
	# padding avec des 0
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# mapping des mots en chiffres pour faire l'aller retour
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# générer une prédiction en fonction d'une entrée
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

# évaluation des performances du modèle
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, eng_tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append(raw_target.split())
		predicted.append(translation.split())
	# calculer le score BLEU
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# chargement des données
dataset = load_clean_sentences('english-french-both.pkl')
train = load_clean_sentences('english-french-train.pkl')
test = load_clean_sentences('english-french-test.pkl')
# préparation du tokenizer anglais
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# préparation du tokenizer français
fra_tokenizer = create_tokenizer(dataset[:, 1])
fra_vocab_size = len(fra_tokenizer.word_index) + 1
fra_length = max_length(dataset[:, 1])
# préparation des données
trainX = encode_sequences(fra_tokenizer, fra_length, train[:, 1])
testX = encode_sequences(fra_tokenizer, fra_length, test[:, 1])

# chargement du modèle pré entraîné
model = load_model('model.h5')
# test sur des données d'entraînement 
print('train')
evaluate_model(model, eng_tokenizer, trainX, train)
# test sur des données de validation 
print('test')
evaluate_model(model, eng_tokenizer, testX, test)