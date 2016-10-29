CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec word2phrase distance word-analogy compute-accuracy

word2vec : src/word2vec.c
	$(CC) $< -o $@ $(CFLAGS)
word2phrase : src/word2phrase.c
	$(CC) $< -o $@ $(CFLAGS)
distance : src/distance.c
	$(CC) $< -o $@ $(CFLAGS)
word-analogy : src/word-analogy.c
	$(CC) $< -o $@ $(CFLAGS)
compute-accuracy : src/compute-accuracy.c
	$(CC) $< -o $@ $(CFLAGS)
	chmod +x *.sh

clean:
	rm -f word2vec word2phrase distance word-analogy compute-accuracy
