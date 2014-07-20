#ifndef PARSE_H
#define PARSE_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 *  Prints help on arguments needed to use the program
*/
void print_usage();

void print_config(int size, int line_size, int assoc, int num_sets);


/*
 *  Parses to cache write policy from cli argument
*/
int parse_write_policy(char* arg);

/*
 *  Parses to cache replacement policy from cli argument
*/
int parse_replacement_policy(char* arg);

const unsigned int maxLineSize = 50;


#endif