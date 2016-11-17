//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <assert.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

#define USE_BLAS 1

#if USE_BLAS
#include "cblas.h"
#endif

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
	long long cn;
	int *point;
	char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING], eval_file[MAX_STRING] = "";
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
char model_type[MAX_STRING];
struct vocab_word *vocab;
int binary = 0,  debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1, batch_size = 500;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
clock_t start;
real *expTable, *final_embeddings;

//For evaluation
long long * analogy_eval_arr;
int nb_analogy_questions = 0;

//TOMOD: Declare model parameters here
//Complex word2vec model
real *word_real, *word_imag, *ctxt_real, *ctxt_imag, *grad_word_real, *grad_word_imag;
//Real word2vec model
real *word_emb, *ctxt_emb, *grad_word_emb, *word_grad_acc, *ctxt_grad_acc;
//Real left right baseline
real *word_right, *word_left, *ctxt_right, *ctxt_left, *grad_word_right, *grad_word_left;
//ENDMOD

int  negative = 5, sign_strat = 0, adagrad = 0;
const int table_size = 1e8, sample_size=5;
const real adagrad_reg = 1e-8;
int *table;


int StartsWith(const char *pre, const char *str) {
	return strncmp(pre, str, strlen(pre)) == 0;
}



void InitUnigramTable() {
	int a, i;
	double train_words_pow = 0;
	double d1, power = 0.75;
	table = (int *)malloc(table_size * sizeof(int));
	for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
	i = 0;
	d1 = pow(vocab[i].cn, power) / train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (double)table_size > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / train_words_pow;
		}
		if (i >= vocab_size) i = vocab_size - 1;
	}
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if (ch == '\n') ungetc(ch, fin);
				break;
			}
			if (ch == '\n') {
				strcpy(word, (char *)"</s>");
				return;
			} else continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1) a--;   // Truncate too long words
	}
	word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
	unsigned int hash = GetWordHash(word);
	while (1) {
		if (vocab_hash[hash] == -1) return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;
	return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = 0;
	vocab_size++;
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size - 1;
	return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
	int a, size;
	unsigned int hash;
	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	size = vocab_size;
	train_words = 0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		if ((vocab[a].cn < min_count) && (a != 0)) {
			vocab_size--;
			free(vocab[a].word);
		} else {
			// Hash will be re-computed, as after the sorting it is not actual
			hash=GetWordHash(vocab[a].word);
			while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = a;
			train_words += vocab[a].cn;
		}
	}
	vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
	// Allocate memory for the binary tree construction
	for (a = 0; a < vocab_size; a++) {
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
	int a, b = 0;
	unsigned int hash;
	for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
		vocab[b].cn = vocab[a].cn;
		vocab[b].word = vocab[a].word;
		b++;
	} else free(vocab[a].word);
	vocab_size = b;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
	min_reduce++;
}

void LearnVocabFromTrainFile() {
	char word[MAX_STRING];
	FILE *fin;
	long long a, i;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	vocab_size = 0;
	AddWordToVocab((char *)"</s>");
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		train_words++;
		if ((debug_mode > 1) && (train_words % 100000 == 0)) {
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		i = SearchVocab(word);
		if (i == -1) {
			a = AddWordToVocab(word);
			vocab[a].cn = 1;
		} else vocab[i].cn++;
		if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
	}
	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	file_size = ftell(fin);
	fclose(fin);
}

void SaveVocab() {
	long long i;
	FILE *fo = fopen(save_vocab_file, "wb");
	for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
	fclose(fo);
}

void ReadVocab() {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin = fopen(read_vocab_file, "rb");
	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	vocab_size = 0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		a = AddWordToVocab(word);
		fscanf(fin, "%lld%c", &vocab[a].cn, &c);
		i++;
	}
	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);
	fclose(fin);
}


void BuildAnalogyEvaluation() {
	FILE *f;
	char st1[MAX_STRING];
	int arr_size = 10000, nb_ignored = 0;
	long long w1_ind, w2_ind, w3_ind, w4_ind;

	long long * tmp;
	analogy_eval_arr = (long long *)malloc(4 * arr_size * sizeof(long long));


	f = fopen(eval_file, "rb");
	
	while (1) {
		ReadWord(st1,f);
		if (feof(f)) break;

		if (! strcmp(st1,":")) {
			ReadWord(st1,f);//Pass the category description
			ReadWord(st1,f); //Line feed
			ReadWord(st1,f); //First word
		}
		//Get all word indexes
		w1_ind = SearchVocab(st1);
		w2_ind = ReadWordIndex(f);
		w3_ind = ReadWordIndex(f);
		w4_ind = ReadWordIndex(f);
		ReadWord(st1,f); //Line feed

		//If one is not found, the anaology question is ignored
		if ( w1_ind == -1 || w2_ind == -1 || w3_ind == -1 || w4_ind == -1 ){
			nb_ignored++;
			continue;
		}

		analogy_eval_arr[nb_analogy_questions * 4] = w1_ind;
		analogy_eval_arr[nb_analogy_questions * 4 + 1] = w2_ind;
		analogy_eval_arr[nb_analogy_questions * 4 + 2] = w3_ind;
		analogy_eval_arr[nb_analogy_questions * 4 + 3] = w4_ind;
		nb_analogy_questions++;

		if ( nb_analogy_questions >= arr_size) { //Reallocate a twice bigger array
			arr_size *= 2;
			tmp  = (long long *)malloc(arr_size * sizeof(long long));
			memcpy(tmp, analogy_eval_arr, (arr_size / 2) * sizeof(long long));
			free(analogy_eval_arr);
			analogy_eval_arr = tmp;
		}
	}
	fclose(f);
	printf("%i/%i analogy questions loaded\n",nb_analogy_questions, nb_ignored + nb_analogy_questions); 
}


void EvalSingleEmbModel( real *embeddings) {

	long long i, j, c, argmax;
	int nb_correct = 0;
	real *emb_copy = (real*) malloc((long long)vocab_size * layer1_size * sizeof(real));
	real *pred_emb = (real*) malloc(layer1_size * sizeof(long long));
	real norm, dot, max;


	//Copying current embeddings (with all concurrent read/write risk implied)
	memcpy(emb_copy, embeddings, (long long)vocab_size * layer1_size * sizeof(real));

	//Normalize embeddings
	for (i = 0; i < vocab_size; i ++) {
		norm = 0;
		for (c = 0; c < layer1_size; c++) {
			norm += emb_copy[i * layer1_size + c] * emb_copy[i * layer1_size + c];
		}
		norm = sqrt(norm);
		for (c = 0; c < layer1_size; c++) {
			emb_copy[i * layer1_size + c] /= norm;
		}
	}

	//Iterate over analogy questions
	for (i = 0; i < nb_analogy_questions; i++ ) {
		//Compute the predicted vector
		for (c = 0; c < layer1_size; c++) {
			pred_emb[c] = - emb_copy[ analogy_eval_arr[i * 4] * layer1_size + c ] 
						+ emb_copy[ analogy_eval_arr[i * 4 + 1] * layer1_size + c ]
						+ emb_copy[ analogy_eval_arr[i * 4 + 2] * layer1_size + c ] ;
		}

		//Find the closest in the vocabulary, TODO: BLAS matrix-vector product should help here
		max = 1.175494e-38;
		argmax = 0;
		for (j = 0; j < vocab_size; j++) {
			dot = 0;
			for (c = 0; c < layer1_size; c++) {
				dot += pred_emb[c] * emb_copy[j * layer1_size + c];
			}
			if (dot > max){
				max = dot;
				argmax = j;
			}
		}
		//If the closest word is the right one
		if (argmax == analogy_eval_arr[i * 4 + 3]){
			nb_correct++;
		}
	}
	printf("Accuracy over the %i analogy questions: %f%%\n",nb_analogy_questions, (float)(nb_correct)/ (float)(nb_analogy_questions) * 100.0);

	free(emb_copy);
	free(pred_emb);

}


void InitNet() {
	long long a, b;
	unsigned long long next_random = 1;

	//TOMOD: Allocate model parameters

	//Complex word2vec model
	if ( StartsWith("complex", model_type)){

		a = posix_memalign((void **)&word_real, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (word_real== NULL) {printf("Memory allocation failed\n"); exit(1);}
		a = posix_memalign((void **)&word_imag, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (word_imag == NULL) {printf("Memory allocation failed\n"); exit(1);}

		a = posix_memalign((void **)&ctxt_real, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (ctxt_real== NULL) {printf("Memory allocation failed\n"); exit(1);}
		a = posix_memalign((void **)&ctxt_imag, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (ctxt_imag== NULL) {printf("Memory allocation failed\n"); exit(1);}

		for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++){
			ctxt_real[a * layer1_size + b] = 0;
			ctxt_imag[a * layer1_size + b] = 0;
		}
		for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
			next_random = next_random * (unsigned long long)25214903917 + 11;
			word_real[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
			word_imag[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
		}
	}

	//Real valued baseline
	if ( StartsWith("2real", model_type)){

		a = posix_memalign((void **)&word_right, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (word_right== NULL) {printf("Memory allocation failed\n"); exit(1);}
		a = posix_memalign((void **)&word_left, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (word_left == NULL) {printf("Memory allocation failed\n"); exit(1);}

		a = posix_memalign((void **)&ctxt_right, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (ctxt_right== NULL) {printf("Memory allocation failed\n"); exit(1);}
		a = posix_memalign((void **)&ctxt_left, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (ctxt_left== NULL) {printf("Memory allocation failed\n"); exit(1);}

		for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++){
			ctxt_right[a * layer1_size + b] = 0;
			ctxt_left[a * layer1_size + b] = 0;
		}
		for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
			next_random = next_random * (unsigned long long)25214903917 + 11;
			word_right[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
			word_left[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
		}
	}

	//Setting order strategy type: right/left context or one word every two
	if (strcmp(model_type, "complex_asym") == 0 || strcmp(model_type, "2real_asym") == 0) {
		sign_strat = 0;
		printf("Asymmetry: right/left context\n");
	} else if (strcmp(model_type, "complex_alt") == 0 || strcmp(model_type, "2real_alt") == 0 ){
		sign_strat = 1;
		printf("Asymmetry: 1 word every 2\n");
	}

	//Real original word2vec model
	if ( StartsWith("real", model_type) ){

		a = posix_memalign((void **)&word_emb, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (word_emb== NULL) {printf("Memory allocation failed\n"); exit(1);}

		a = posix_memalign((void **)&ctxt_emb, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (ctxt_emb== NULL) {printf("Memory allocation failed\n"); exit(1);}

		if (adagrad) {
			a = posix_memalign((void **)&word_grad_acc, 128, (long long)vocab_size * layer1_size * sizeof(real));
			if (word_grad_acc== NULL) {printf("Memory allocation failed\n"); exit(1);}
			a = posix_memalign((void **)&ctxt_grad_acc, 128, (long long)vocab_size * layer1_size * sizeof(real));
			if (ctxt_grad_acc== NULL) {printf("Memory allocation failed\n"); exit(1);}
		}

		for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++){
			ctxt_emb[a * layer1_size + b] = 0;
		}
		for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
			next_random = next_random * (unsigned long long)25214903917 + 11;
			word_emb[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
		}
	}

	//ENMOD
}


//Builds next batch of training pairs. Emulate a python-style yield.
void* BuildNextBatch(long long *batch, long long *a,long long *b,long long *d, long long *word_count, long long *last_word_count, 
		long long *word, long long *last_word, long long *sentence_length, long long *sentence_position, long long *sen, long long *local_iter,
		unsigned long long *next_random, clock_t * now, FILE* fi, void* id) {

	long long i = 0, c = 0, target, label;

	while (1) {
		if (*a == *b) { //Else jumps back to where we were

			if (*word_count - *last_word_count > 10000) {
				word_count_actual += *word_count - *last_word_count;
				*last_word_count = *word_count;
				if ((debug_mode > 1)) {
					*now=clock();
					printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
							word_count_actual / (real)(iter * train_words + 1) * 100,
							word_count_actual / ((real)(*now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
					fflush(stdout);
				}
				if (!adagrad) alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
				if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
			}

			if (*sentence_length == 0) {
				while (1) {
					*word = ReadWordIndex(fi);
					if (feof(fi)) break;
					if (*word == -1) continue;
					(*word_count)++;
					if (*word == 0) break;
					// The subsampling randomly discards frequent words while keeping the ranking same
					if (sample > 0) {
						real ran = (sqrt(vocab[*word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[*word].cn;
						*next_random = *next_random * (unsigned long long)25214903917 + 11;
						if (ran < (*next_random & 0xFFFF) / (real)65536) continue;
					}
					sen[*sentence_length] = *word;
					(*sentence_length)++;
					if (*sentence_length >= MAX_SENTENCE_LENGTH) break;
				}
				*sentence_position = 0;
			}

			if (feof(fi) || (*word_count > train_words / num_threads)) {
				word_count_actual += *word_count - *last_word_count;
				(*local_iter)--;
				*word_count = 0;
				*last_word_count = 0;
				*sentence_length = 0;
				fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
				//Run evaluation at each epoch for one thread only
				if (id == 0 && StartsWith("real_original", model_type) && strlen(eval_file) > 0) {
					EvalSingleEmbModel(word_emb);
				}
				continue;
			}
			*word = sen[*sentence_position];
			if (*word == -1) continue;
		}

		//That random 'b' starting point makes the window size not constant, but uniformly distributed in [0,window]
		//Makes sense, as closer context is probably more linked to target word.
		//Maybe uniform is not even enough, maybe we should make it geometrically decreasing with the distance to the target word
		while ( *a < window * 2 + 1 - *b) {
			if (*a != window) {
				if (*d == 0){ //Else jumps back where we were
					c = *sentence_position - window + *a;
					if (c < 0) { (*a)++; continue; }
					if (c >= *sentence_length) { (*a)++; continue; }
					*last_word = sen[c];
					if (*last_word == -1) { (*a)++; continue; }
				}

				// NEGATIVE SAMPLING
				while ( *d < negative + 1) {
					if (*d == 0) {
						target = *word;
						label = 1;
					} else {
						*next_random = *next_random * (unsigned long long)25214903917 + 11;
						target = table[(*next_random >> 16) % table_size];
						if (target == 0) target = *next_random % (vocab_size - 1) + 1;
						if (target == *word) { (*d)++; continue; }
						label = 0;
					}

					//Storing the batch indexes and the order to consider
					batch[i*sample_size] = *last_word;
					batch[i*sample_size+1] = target;
					batch[i*sample_size+2] = label;

					//TOMOD: Sign of the imaginary part: 
					//1: differentiates right and left contexts
					//2: one word every two
					if (sign_strat == 0) {
						batch[i*sample_size+3] = (*a > window) * 2 - 1; 
					} else if ( sign_strat == 1 ) {
						if (*a < window)	batch[i*sample_size+3] = ((*a - *b) % 2) * 2 - 1; 
						if (*a > window)	batch[i*sample_size+3] = ((*a - *b + 1) % 2) * 2 - 1; 
					}
					//ENDMOD

					//Controlling word gradient updates:
					if (*d == negative){
						batch[i*sample_size+4] = (long long)1;
					} else {
						batch[i*sample_size+4] = (long long)0;
					}
					
					(*d)++;
					i++; if (i == batch_size) return 0;
				}
				*d = 0; //Reinit for next loop
			
			}
			(*a)++;
		}
		*next_random = *next_random * (unsigned long long)25214903917 + 11;
		*b = *next_random % window;
		*a = *b; //Reinit for next loop


		(*sentence_position)++;
		if (*sentence_position >= *sentence_length) {
			*sentence_length = 0;
			continue;
		}
	}
}


//////////////////////////////////////////////////////////////////////////////////
// REAL MODEL
//////////////////////////////////////////////////////////////////////////////////


void *TrainRealModelThread(void *id) {
	//Data processing variables
	long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
	long long l1, l2, i, c, target, label, local_iter = iter, update_word_embs;
	unsigned long long next_random = (long long)id;
	clock_t now;
	long long *batch = (long long *)calloc(sample_size * batch_size, sizeof(long long));
	FILE *fi = fopen(train_file, "rb");
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	//Init variables for batch generation
	next_random = next_random * (unsigned long long)25214903917 + 11;
	b = next_random % window;
	a = b;
	d = 0;


	//TOMOD: Model variables
	real f, g, tmp_grad;
	real *grad_word_emb = (real *)calloc(layer1_size, sizeof(real));
	//Init gradient accumulators:
	for (c = 0; c < layer1_size; c++) grad_word_emb[c] = 0;

	//If we're using unique embeddings for word/context, simply redirect the ctxt pointer:
	if ( strcmp(model_type, "real_unique") == 0 ){
		free(ctxt_emb);
		ctxt_emb = word_emb;
	}	
	//ENDMOD

	while (1) {
		//Create the next batch
		BuildNextBatch(batch, &a, &b, &d, &word_count, &last_word_count, &word, &last_word, &sentence_length, &sentence_position, sen, &local_iter, &next_random, &now, fi, id);

		if (local_iter == 0) break;

		for (i = 0; i < batch_size; i++) {
			//train skip-gram
			last_word = batch[i*sample_size];
			target = batch[i*sample_size + 1];
			label = batch[i*sample_size + 2];
			update_word_embs = batch[i*sample_size + 4];

			l1 = last_word * layer1_size;
			l2 = target * layer1_size;

			

			//TOMOD: Gradient computations and updates
			//Computing score
#if USE_BLAS
			f = cblas_sdot(layer1_size, word_emb + l1 , 1, ctxt_emb + l2 , 1);
#else 
			f = 0;
			for (c = 0; c < layer1_size; c++){
				f += word_emb[c + l1] * ctxt_emb[c + l2];
			}
#endif

			if (f > MAX_EXP) g = (label - 1);
			else if (f < -MAX_EXP) g = (label - 0);
			else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]);

#if 0//USE_BLAS //Slower so set to zero

			//Computing word gradients (use neue1e as tmp vector for vectorization)
			cblas_saxpy(layer1_size, g, ctxt_emb + l2, 1, grad_word_emb, 1);
			//Computing context gradients
			cblas_saxpy(layer1_size, g, word_emb + l1, 1, ctxt_emb + l2, 1);
#else 
			if ( adagrad ) {
				for (c = 0; c < layer1_size; c++){
					//Computing word gradients
					tmp_grad = g * ctxt_emb[c + l2] ;
					word_grad_acc[c + l1] += tmp_grad * tmp_grad;
					grad_word_emb[c] += (alpha / (sqrt( word_grad_acc[c + l1]) + adagrad_reg)) * tmp_grad;
					//Computing context gradients & updating embeddings
					tmp_grad = g * word_emb[c + l1];
					ctxt_grad_acc[c + l2] += tmp_grad * tmp_grad;
					ctxt_emb[c + l2] += (alpha / (sqrt( ctxt_grad_acc[c + l2]) + adagrad_reg)) * tmp_grad;
				}
			} else {
				g *= alpha;
				for (c = 0; c < layer1_size; c++){
					//Computing word gradients
					grad_word_emb[c] += g * ctxt_emb[c + l2] ;
					//Computing context gradients & updating embeddings
					ctxt_emb[c + l2] += g * word_emb[c + l1] ;
				}
			}

#endif
			if (update_word_embs == 1){
#if 0//USE_BLAS //Slower so set to zero
				cblas_saxpy(layer1_size, 1, grad_word_emb, 1, word_emb + l1, 1);
				for (c = 0; c < layer1_size; c++) grad_word_emb[c] = 0;
#else
				for (c = 0; c < layer1_size; c++){
					//Updating word embeddings
					word_emb[c + l1] += grad_word_emb[c];
					//Resetting gradient accumulator
					grad_word_emb[c] = 0;
				}
#endif
			}
			//ENDMOD
		}
	}
	fclose(fi);
	//TOMOD: Free local vectors
	free(grad_word_emb);
	//ENDMOD
	pthread_exit(NULL);
}



//////////////////////////////////////////////////////////////////////////////////
// REAL BASELINE LEFT RIGHT MODEL
//////////////////////////////////////////////////////////////////////////////////


void *TrainRealBaselineModelThread(void *id) {
	//Data processing variables
	long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
	long long l1, l2, i, c, target, label, local_iter = iter, update_word_embs;
	unsigned long long next_random = (long long)id;
	clock_t now;
	long long *batch = (long long *)calloc(sample_size * batch_size, sizeof(long long));
	FILE *fi = fopen(train_file, "rb");
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	//Init variables for batch generation
	next_random = next_random * (unsigned long long)25214903917 + 11;
	b = next_random % window;
	a = b;
	d = 0;


	//TOMOD: Model variables
	real f, g, order_sign;
	real *cur_word_emb, *cur_ctxt_emb, *cur_grad_word;
	real *grad_word_right = (real *)calloc(layer1_size, sizeof(real));
	real *grad_word_left = (real *)calloc(layer1_size, sizeof(real));
	//Init gradient accumulators:
	for (c = 0; c < layer1_size; c++) grad_word_right[c] = 0;
	for (c = 0; c < layer1_size; c++) grad_word_left[c] = 0;
	//ENDMOD



	while (1) {
		//Create the next batch
		BuildNextBatch(batch, &a, &b, &d, &word_count, &last_word_count, &word, &last_word, &sentence_length, &sentence_position, sen, &local_iter, &next_random, &now, fi, id);

		if (local_iter == 0) break;

/*
		for (i = 0; i < batch_size; i++) {
			last_word = batch[i*sample_size];
			target = batch[i*sample_size + 1];
			label = batch[i*sample_size + 2];
			imag_part_sign = (real) batch[i*sample_size + 3];
			update_word_embs = batch[i*sample_size + 4];
			printf("%i\t%i\t%i\t%f\t%i\n",last_word,target,label,imag_part_sign,update_word_embs);
		}
		exit(0);
*/

		for (i = 0; i < batch_size; i++) {
			//train skip-gram
			last_word = batch[i*sample_size];
			target = batch[i*sample_size + 1];
			label = batch[i*sample_size + 2];
			order_sign = batch[i*sample_size + 3];
			update_word_embs = batch[i*sample_size + 4];

			l1 = last_word * layer1_size;
			l2 = target * layer1_size;

			
			//TOMOD: Gradient computations and updates
			if (order_sign == 1){
				cur_word_emb = word_right + l1;
				cur_ctxt_emb = ctxt_right + l2;
				cur_grad_word = grad_word_right; 
			} else {
				cur_word_emb = word_left + l1;
				cur_ctxt_emb = ctxt_left + l2;
				cur_grad_word = grad_word_left; 
			}

			//Computing score
#if USE_BLAS
			f = cblas_sdot(layer1_size, cur_word_emb, 1, cur_ctxt_emb, 1);
#else 
			f = 0;
			for (c = 0; c < layer1_size; c++){
				f += cur_word_emb[c] * cur_ctxt_emb[c];
			}
#endif

			if (f > MAX_EXP) g = (label - 1) * alpha;
			else if (f < -MAX_EXP) g = (label - 0) * alpha;
			else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

#if 0//USE_BLAS //Slower so set to zero

			//Computing word gradients (use neue1e as tmp vector for vectorization)
			cblas_saxpy(layer1_size, g, cur_ctxt_emb, 1, cur_grad_word, 1);
			//Computing context gradients
			cblas_saxpy(layer1_size, g, cur_word_emb, 1, cur_ctxt_emb, 1);
#else 
			for (c = 0; c < layer1_size; c++){
				//Computing word gradients
				cur_grad_word[c] += g * cur_ctxt_emb[c] ;
				//Computing context gradients & updating embeddings
				cur_ctxt_emb[c] += g * cur_word_emb[c] ;
			}

#endif
			if (update_word_embs == 1){
#if 0//USE_BLAS //Slower so set to zero
				cblas_saxpy(layer1_size, 1, grad_word_right, 1, word_right + l1, 1);
				cblas_saxpy(layer1_size, 1, grad_word_left, 1, word_left + l1, 1);
				for (c = 0; c < layer1_size; c++) grad_word_right[c] = 0;
				for (c = 0; c < layer1_size; c++) grad_word_left[c] = 0;
#else
				for (c = 0; c < layer1_size; c++){
					//Updating word embeddings
					word_right[c + l1] += grad_word_right[c];
					word_left[c + l1] += grad_word_left[c];
					//Resetting gradient accumulator
					grad_word_right[c] = 0;
					grad_word_left[c] = 0;
				}
#endif
			}
			//ENDMOD
		}
	}
	fclose(fi);
	//TOMOD: Free local vectors
	free(grad_word_right);
	free(grad_word_left);
	//ENDMOD
	pthread_exit(NULL);
}



//////////////////////////////////////////////////////////////////////////////////
// COMPLEX MODEL
//////////////////////////////////////////////////////////////////////////////////


void *TrainComplexModelThread(void *id) {
	//Data processing variables
	long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
	long long l1, l2, i, c, target, label, local_iter = iter, update_word_embs;
	unsigned long long next_random = (long long)id;
	clock_t now;
	long long *batch = (long long *)calloc(sample_size * batch_size, sizeof(long long));
	FILE *fi = fopen(train_file, "rb");
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	//Init variables for batch generation
	next_random = next_random * (unsigned long long)25214903917 + 11;
	b = next_random % window;
	a = b;
	d = 0;


	//TOMOD: Model variables
	real f, g, imag_part_sign, dot_real, dot_imag;
	real *tmp_vect = (real *)calloc(layer1_size, sizeof(real));
	real *grad_word_real = (real *)calloc(layer1_size, sizeof(real));
	real *grad_word_imag = (real *)calloc(layer1_size, sizeof(real));
	//Init gradient accumulators:
	for (c = 0; c < layer1_size; c++) grad_word_real[c] = 0;
	for (c = 0; c < layer1_size; c++) grad_word_imag[c] = 0;
	//ENDMOD



	while (1) {
		//Create the next batch
		BuildNextBatch(batch, &a, &b, &d, &word_count, &last_word_count, &word, &last_word, &sentence_length, &sentence_position, sen, &local_iter, &next_random, &now, fi, id);

		if (local_iter == 0) break;

/*
		for (i = 0; i < batch_size; i++) {
			last_word = batch[i*sample_size];
			target = batch[i*sample_size + 1];
			label = batch[i*sample_size + 2];
			imag_part_sign = (real) batch[i*sample_size + 3];
			update_word_embs = batch[i*sample_size + 4];
			printf("%i\t%i\t%i\t%f\t%i\n",last_word,target,label,imag_part_sign,update_word_embs);
		}
		exit(0);
*/

		for (i = 0; i < batch_size; i++) {
			//train skip-gram
			last_word = batch[i*sample_size];
			target = batch[i*sample_size + 1];
			label = batch[i*sample_size + 2];
			imag_part_sign = batch[i*sample_size + 3];
			update_word_embs = batch[i*sample_size + 4];

			l1 = last_word * layer1_size;
			l2 = target * layer1_size;

			

			//TOMOD: Gradient computations and updates
			//Computing score
#if USE_BLAS
			dot_real = cblas_sdot(layer1_size, word_real + l1 , 1, ctxt_real + l2 , 1);
			dot_real += cblas_sdot(layer1_size, word_imag + l1 , 1, ctxt_imag + l2 , 1);
			dot_imag = cblas_sdot(layer1_size, word_real + l1 , 1, ctxt_imag + l2 , 1);
			dot_imag -= cblas_sdot(layer1_size, word_imag + l1 , 1, ctxt_real + l2 , 1);
#else 
			dot_real = 0; dot_imag = 0;
			for (c = 0; c < layer1_size; c++){
				dot_real += word_real[c + l1] * ctxt_real[c + l2] + word_imag[c + l1] * ctxt_imag[c + l2];
				dot_imag += word_real[c + l1] * ctxt_imag[c + l2] - word_imag[c + l1] * ctxt_real[c + l2];
			}
#endif
			//Order is taken into account with the sign value in 'imag_part_sign'
			f = dot_real + imag_part_sign * dot_imag;

			if (f > MAX_EXP) g = (label - 1) * alpha;
			else if (f < -MAX_EXP) g = (label - 0) * alpha;
			else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

#if 0//USE_BLAS //Slower so set to zero

			//Computing word gradients (use neue1e as tmp vector for vectorization)
			cblas_scopy(layer1_size, ctxt_real + l2, 1, tmp_vect, 1);
			cblas_saxpy(layer1_size, imag_part_sign, ctxt_imag + l2, 1, tmp_vect, 1);
			cblas_saxpy(layer1_size, g, tmp_vect, 1, grad_word_real, 1);
			cblas_scopy(layer1_size, ctxt_imag + l2, 1, tmp_vect, 1);
			cblas_saxpy(layer1_size, -imag_part_sign, ctxt_real + l2, 1, tmp_vect, 1);
			cblas_saxpy(layer1_size, g, tmp_vect, 1, grad_word_imag, 1);
			//Computing context gradients
			cblas_scopy(layer1_size, word_real + l1, 1, tmp_vect, 1);
			cblas_saxpy(layer1_size, -imag_part_sign, word_imag + l1, 1, tmp_vect, 1);
			cblas_saxpy(layer1_size, g, tmp_vect, 1, ctxt_real + l2, 1);
			cblas_scopy(layer1_size, word_imag + l1, 1, tmp_vect, 1);
			cblas_saxpy(layer1_size, imag_part_sign, word_real + l1, 1, tmp_vect, 1);
			cblas_saxpy(layer1_size, g, tmp_vect, 1, ctxt_imag + l2, 1);
#else 
			for (c = 0; c < layer1_size; c++){
				//Computing word gradients
				grad_word_real[c] += g * ( ctxt_real[c + l2] + imag_part_sign * ctxt_imag[c + l2])  ;
				grad_word_imag[c] += g * ( ctxt_imag[c + l2] - imag_part_sign * ctxt_real[c + l2])  ;
				//Computing context gradients & updating embeddings
				ctxt_real[c + l2] += g * ( word_real[c + l1] - imag_part_sign * word_imag[c + l1] ) ;
				ctxt_imag[c + l2] += g * ( word_imag[c + l1] + imag_part_sign * word_real[c + l1] ) ;
				
			}

#endif
			if (update_word_embs == 1){
#if 0//USE_BLAS //Slower so set to zero

				cblas_saxpy(layer1_size, 1, grad_word_real, 1, word_real + l1, 1);
				cblas_saxpy(layer1_size, 1, grad_word_imag, 1, word_imag + l1, 1);
				for (c = 0; c < layer1_size; c++) grad_word_real[c] = 0;
				for (c = 0; c < layer1_size; c++) grad_word_imag[c] = 0;
#else
				for (c = 0; c < layer1_size; c++){
					//Updating word embeddings
					word_real[c + l1] += grad_word_real[c];
					word_imag[c + l1] += grad_word_imag[c];
					//Resetting gradient accumulator
					grad_word_real[c] = 0;
					grad_word_imag[c] = 0;
				}
#endif
			}
			//ENDMOD
		}
	}
	fclose(fi);
	//TOMOD: Free local vectors
	free(tmp_vect);
	free(grad_word_real);
	free(grad_word_imag);
	//ENDMOD
	pthread_exit(NULL);
}



void TrainModel() {
	long a, b, c, d;
	FILE *fo;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", train_file);
	starting_alpha = alpha;
	if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
	if (save_vocab_file[0] != 0) SaveVocab();
	if (output_file[0] == 0) return;
	if (strlen(eval_file) > 0) BuildAnalogyEvaluation();
	InitNet();
	if (negative > 0) InitUnigramTable();
	start = clock();
	//TOMOD: Starts threads on the corresponding model function
	if ( StartsWith("complex", model_type)){
		for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainComplexModelThread, (void *)a);
		for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	} else if ( StartsWith("2real", model_type)){
		for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainRealBaselineModelThread, (void *)a);
		for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	} else if ( StartsWith("real", model_type)) {
		for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainRealModelThread, (void *)a);
		for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	}
	//ENMOD

	fo = fopen(output_file, "wb");
	if (classes == 0) {
		// Save the word vectors
		//TOMOD: Chose how to save the embeddings
		if ( StartsWith("complex", model_type)){
			fprintf(fo, "%lld %lld\n", vocab_size, 2*layer1_size); //2 times to concatenate real and imaginary parts
			for (a = 0; a < vocab_size; a++) {
				fprintf(fo, "%s ", vocab[a].word);
				if (binary){
					for (b = 0; b < layer1_size; b++){
						fwrite(&word_real[a * layer1_size + b], sizeof(real), 1, fo);
						fwrite(&word_imag[a * layer1_size + b], sizeof(real), 1, fo);
					}
				}
				else {
					for (b = 0; b < layer1_size; b++) {
						fprintf(fo, "%lf ", word_real[a * layer1_size + b]);
						fprintf(fo, "%lf ", word_imag[a * layer1_size + b]);
					}
				}
				fprintf(fo, "\n");
			}
		} else if ( StartsWith("2real", model_type)){
			fprintf(fo, "%lld %lld\n", vocab_size, 2*layer1_size); //2 times to concatenate real and imaginary parts
			for (a = 0; a < vocab_size; a++) {
				fprintf(fo, "%s ", vocab[a].word);
				if (binary){
					for (b = 0; b < layer1_size; b++){
						fwrite(&word_right[a * layer1_size + b], sizeof(real), 1, fo);
						fwrite(&word_left[a * layer1_size + b], sizeof(real), 1, fo);
					}
				}
				else {
					for (b = 0; b < layer1_size; b++) {
						fprintf(fo, "%lf ", word_right[a * layer1_size + b]);
						fprintf(fo, "%lf ", word_left[a * layer1_size + b]);
					}
				}
				fprintf(fo, "\n");
			}
		} else if (StartsWith("real",model_type)) {
			fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
			for (a = 0; a < vocab_size; a++) {
				fprintf(fo, "%s ", vocab[a].word);
				if (binary){
					for (b = 0; b < layer1_size; b++){
						fwrite(&word_emb[a * layer1_size + b], sizeof(real), 1, fo);
					}
				}
				else {
					for (b = 0; b < layer1_size; b++) {
						fprintf(fo, "%lf ", word_emb[a * layer1_size + b]);
					}
				}
				fprintf(fo, "\n");
			}
		}
		//ENMOD
	} else {
		// Run K-means on the word vectors
		int clcn = classes, iter = 10, closeid;
		int *centcn = (int *)malloc(classes * sizeof(int));
		int *cl = (int *)calloc(vocab_size, sizeof(int));
		real closev, x;
		real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
		for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
		for (a = 0; a < iter; a++) {
			for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
			for (b = 0; b < clcn; b++) centcn[b] = 1;
			for (c = 0; c < vocab_size; c++) {
				for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += word_emb[c * layer1_size + d];
				centcn[cl[c]]++;
			}
			for (b = 0; b < clcn; b++) {
				closev = 0;
				for (c = 0; c < layer1_size; c++) {
					cent[layer1_size * b + c] /= centcn[b];
					closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
				}
				closev = sqrt(closev);
				for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
			}
			for (c = 0; c < vocab_size; c++) {
				closev = -10;
				closeid = 0;
				for (d = 0; d < clcn; d++) {
					x = 0;
					for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * word_emb[c * layer1_size + b];
					if (x > closev) {
						closev = x;
						closeid = d;
					}
				}
				cl[c] = closeid;
			}
		}
		// Save the K-means classes
		for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
		free(centcn);
		free(cent);
		free(cl);
	}
	fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
		printf("\t-eval <file>\n");
		printf("\t\tUse the analogy questions from <file> to produce evaluation every epoch\n");
		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");
		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");
		printf("\t-sample <float>\n");
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
		printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 12)\n");
		printf("\t-iter <int>\n");
		printf("\t\tRun more training iterations (default 5)\n");
		printf("\t-min-count <int>\n");
		printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025 (skip-gram)\n");
		printf("\t-classes <int>\n");
		printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		printf("\t-save-vocab <file>\n");
		printf("\t\tThe vocabulary will be saved to <file>\n");
		printf("\t-model <name>\n");
		printf("\t\tThe model to use, possible value are 'complex_asym', 'complex_alt', 'real_original', 'real_unique', '2real_asym', '2real_alt'\n");
		printf("\t-adagrad <int>\n");
		printf("\t\tActivates adagrad learning step if non-zero. Only for the 'real_original' model for the moment.\n");
		printf("\t-read-vocab <file>\n");
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\nExamples:\n");
		printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -binary 0 -iter 3\n\n");
		return 0;
	}
	output_file[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-eval", argc, argv)) > 0) strcpy(eval_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-model", argc, argv)) > 0) strcpy(model_type, argv[i + 1]);
	if ((i = ArgPos((char *)"-adagrad", argc, argv)) > 0) adagrad = atoi(argv[i + 1]);

	//TOMOD; Add model string id
	if (! (strcmp(model_type, "complex_alt") == 0 || strcmp(model_type, "complex_asym") == 0
		|| strcmp(model_type, "2real_alt") == 0 || strcmp(model_type, "2real_asym") == 0
		|| strcmp(model_type, "real_unique") == 0 || strcmp(model_type, "real_unique") == 0
		|| strcmp(model_type, "real_original") == 0 )) {
		printf("Model type '%s' unknown, choices are: 'complex_asym', 'complex_alt', 'complex_symm', 'real_original', '2real_asym', '2real_alt', 'real_unique'.\n", model_type);
	//ENDMOD
		exit(1);
	}

	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}
	TrainModel();
	return 0;
}
