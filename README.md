# Hidden-Markov-Models
Using Hidden Markov Models (HMM) and Part-of-speech (POS) tagging, the most probable meaning of the sentence "Flies fly" was found.

Nltk's packages of the Brown subcorpus and universal tagsets were used to calculate emission, transition, and starting proababilities. These probabilities were then put through the Viterbi algorithm to find that "Flies fly" is most likely Flies (Noun) followed by fly (Verb), as opposed to any other combination of POS (Preposition, Adverb, etc.)
