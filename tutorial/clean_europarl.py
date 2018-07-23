# importing the dependencies
import re
import string
from pickle
from unicodedata import normalize
from collections import Counter

# loading, cleaning and saving the eruoparl dataset

def load_data(filename):
	# loading of data
	file = open(filename, mode = 'rt', encoding = 'utf-8') # opening the file
	text = file.read() # read the data
	file.close() # close the file
	return text

def to_sentences(text):
	# convert to sentences
	return text.strip().split('\n')

def sentence_length(sentences):
	# get sentence length
	lenghts = [len(s.split()) for s in sentences]
	return min(lenghts), max(lenghts)

def clean_lines(lines):
	# clean a list of lines
	cleaned = []
	# prepare regex for character filtering
	re_print = re.compile('[^{0}]'.format(re.escape(string.printable)))
	# prepare tranaslation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# normalize unicode characters
		line = normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [word.translate(table) for word in line]
		# remove non-printable chars form each token
		line = [re_print.sub('', w) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	return cleaned

def to_vocab(lines):
	vocab = Counter()
	for line in lines:
		tokens = line.split()
		vocab.update(tokens)
	return vocab

# remove all the words below a certain frequency
def trim_vocab(vocab, min_occurance):
	tokens = [k for k, c in vocab.items() if c >= min_occurance]
	return tokens

def update_dataset(lines, vocab):
	new_lines = []
	for line in lines:
		new_tokens = []
		for token in line.split():
			if token in vocab:
				new_tokens.append(token)
			else:
				new_tokens.append('<UNK>')
		new_line = ' '.join(new_tokens)
		new_lines.append(new_line)
	return new_lines

# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
	pickle.dump(sentences, open(filename, 'wb'))
	print('Saved: {0}'.format(filename))

# load the english dataset
print('Processing ')
filename = 'europarl-v7.fr-en.en'
doc = load_data(filename)
sentences = to_sentences(doc)
minlen_eng, maxlen_eng = sentence_length(sentences)
print('English dataset:\nsentences: {0}\nmin: {1}, max: {2}'.format(len(sentences), minlen_eng, maxlen_eng))
sentences = clean_lines(sentences)
vocab = to_vocab(sentences) # convert to tokens
print('vocab before cleaning:', len(vocab))
vocab = trim_vocab(vocab, 5) # min_occurance = 5
print('vocab after cleaning:', len(vocab))
lines = update_dataset(sentences, vocab)
save_clean_sentences(lines, 'europarl_english.pkl')
# check
for i in range(10):
	print(sentences[i])

# load the french dataset
filename = 'europarl-v7.fr-en.fr'
doc = load_data(filename)
sentences = to_sentences(doc)
minlen_fr, maxlen_fr = sentence_length(sentences)
print('French dataset:\nsentences: {0}\nmin: {1}, max: {2}'.format(len(sentences), minlen_fr, maxlen_fr))
sentences = clean_lines(sentences)
vocab = to_vocab(sentences) # convert to tokens
print('vocab before cleaning:', len(vocab))
vocab = trim_vocab(vocab, 5) # min_occurance = 5
print('vocab after cleaning:', len(vocab))
lines = update_dataset(sentences, vocab)
save_clean_sentences(lines, 'europarl_french.pkl')
# check
for i in range(10):
	print(lines[i])

# util function can be used later
def load_clean_sentences(filename):
	return pickle.load(open(filename, 'rb'))

