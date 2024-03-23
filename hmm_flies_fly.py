"""HMM_flies_fly
*   Aarifah Ullah
*   Prof. Kasia Hitczenko
*   CSCI 4907.82 Natural Language Understanding
*   26 February 2024
"""

import nltk
from nltk.corpus import brown
from nltk.probability import FreqDist
import numpy as np

nltk.download('brown')
nltk.download('universal_tagset')

# Calculate starting probabilities
# This is provided in the homework; no need to change this

brown_sents = brown.tagged_sents(tagset='universal')
starting_pos = [sent[0][1] for sent in brown_sents]
freq_pos = FreqDist(starting_pos)
for item in freq_pos:
  freq_pos[item] = freq_pos[item] / len(starting_pos)

#Calculate emission probabilities, case insensitive
#Remember to ignore case and treat "Flies" the same as "flies"
#emission probability = P('fly'|VERB), meaning how likely the POS class VERB will pick 'fly' as one of is examples out of all possible words that are also labelled as VERBS

from collections import defaultdict, Counter

brown_words = brown.tagged_words(tagset='universal')
word_tags_counts = defaultdict(Counter) #track words with their POS & how many times it shows up with that POS
for word, pos in brown_words:
  word_tags_counts[word][pos] += 1

#the associated part of speech tags for "flies" and "fly" (case-insensitive)
print("POS Tag sets:")
print('flies', word_tags_counts['flies'])
print('Flies', word_tags_counts['Flies'])
print('fly', word_tags_counts['fly'])
print('Fly', word_tags_counts['Fly'])

verb_count = 0
noun_count = 0
#iterate through list of words
for word in word_tags_counts:
  if "VERB" in word_tags_counts[word]: #count how many words are classified as a VERB
    verb_count += 1
  if "NOUN" in word_tags_counts[word]: #count how many words are classified as a NOUN
    noun_count += 1

#flies/Flies emission probability:
flies_noun_count = word_tags_counts['flies'].get("NOUN") + word_tags_counts['Flies'].get("NOUN") #returns 8
flies_verb_count = word_tags_counts['flies'].get("VERB")
flies_given_noun_probability = flies_noun_count / noun_count #emission proabability P('flies/Flies'|NOUN)
flies_given_verb_probability = flies_verb_count / verb_count # P('flies/Flies'|VERB)
print("'flies' emission probabilities:")
print("P('flies'|NOUN) = ", flies_given_noun_probability)
print("P('flies'|VERB) = ", flies_given_verb_probability)

#fly/Fly emission probability:
fly_noun_count = word_tags_counts['fly'].get("NOUN") + word_tags_counts['Fly'].get("NOUN") #returns 15
fly_verb_count = word_tags_counts['fly'].get("VERB")
fly_given_noun_probability = fly_noun_count / noun_count #emission proabability P('fly/Fly'|NOUN)
fly_given_verb_probability = fly_verb_count / verb_count # P('fly/Fly'|VERB)
print("'fly' emission probabilities:")
print("P('fly'|NOUN) = ", fly_given_noun_probability)
print("P('fly'|VERB) = ", fly_given_verb_probability)

#Calculate transition probabilities
word_tag_pairs = list(nltk.bigrams(brown_words))

verb_given_verb_count = 0 #[0,0]
noun_given_verb_count = 0 #[0,1]
verb_given_noun_count = 0 #[1,0]
noun_given_noun_count = 0 #[1,1]

all_given_verb_count = 0 #count all POS instance pairs for verbs
all_given_noun_count = 0 #count all POS instance pairs for nouns

for ((word1,pos1),(word2,pos2)) in word_tag_pairs: #iterate through all pairs of consecutive words / POS tags
  if((pos1 == "VERB") & (pos2 == "VERB")):
    verb_given_verb_count += 1 #returns 33672
  if((pos1 == "VERB") & (pos2 == "NOUN")):
    noun_given_verb_count += 1 #returns 17851
  if((pos1 == "VERB")):
    all_given_verb_count += 1 #returns 182750
  if((pos1 == "NOUN") & (pos2 == "VERB")):
    verb_given_noun_count += 1 #returns 43819
  if((pos1 == "NOUN") & (pos2 == "NOUN")):
    noun_given_noun_count += 1 #returns 41309
  if((pos1 == "NOUN")):
    all_given_noun_count += 1 #returns 275558

#final transmission probabilities are the ratios between a specific count over general count
verb_given_verb = verb_given_verb_count / all_given_verb_count
noun_given_verb = noun_given_verb_count / all_given_verb_count
verb_given_noun = verb_given_noun_count / all_given_noun_count
noun_given_noun = noun_given_noun_count / all_given_noun_count

print("transmission probabilities:")
print("P(VERB|VERB) = ",verb_given_verb)
print("P(NOUN|VERB) = ",noun_given_verb)
print("P(VERB|NOUN) = ",verb_given_noun)
print("P(NOUN|NOUN) = ",noun_given_noun)

#the Virterbi algorithm was done by hand after finding all needed probabilities.
