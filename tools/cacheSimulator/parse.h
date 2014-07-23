#ifndef PARSE_H
#define PARSE_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <list>
#include <tuple>

#include "common.h"

typedef std::vector<std::tuple<std::vector<unsigned int>,unsigned int,std::list<Entry>>>  TRACE_VEC;


TRACE_VEC parse(std::ifstream& input);


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


#endif