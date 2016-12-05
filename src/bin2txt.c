#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

const long long max_size = 2000;         // max length of strings
const long long max_w = 50;              // max length of vocabulary entries


//Write in format readable by this evaluation script: https://github.com/mfaruqui/eval-word-vectors
int main(int argc, char **argv) {
	FILE *f, *fo;
	char out_file[max_size], in_file[max_size], cur_word[max_w];
	float cur_val;
	long long words, size, a, b;
	if (argc < 2) {
		printf("Usage: ./bin2txt <IN_FILE> <OUT_FILE>\n");
		return 0;
	}
	strcpy(in_file, argv[1]);
	strcpy(out_file, argv[2]);
	f = fopen(in_file, "rb");
	fo = fopen(out_file, "wb");
	if (f == NULL) {
		printf("Input file not found\n");
		return -1;
	}

	fscanf(f, "%lld", &words);
	fscanf(f, "%lld", &size);
	fprintf(fo, "%lld %lld\n", words, size);

	for (b = 0; b < words; b++) {
		a = 0;
		while (1) {
			cur_word[a] = fgetc(f);
			if (feof(f) || (cur_word[a] == ' ')) break;
			if ((a < max_w) && (cur_word[a] != '\n')) a++;
		}
		cur_word[a] = 0;
		fprintf(fo, "%s ", cur_word);
		for (a = 0; a < size; a++){
			fread(&cur_val, sizeof(float), 1, f);
			fprintf(fo, "%lf ", cur_val);
		}
		fprintf(fo, "\n");
	}
	fclose(f);
	fclose(fo);
	return 0;
}
