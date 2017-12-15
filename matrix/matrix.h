#pragma once
#include <iostream>
#include <time.h>
#include <omp.h>
#define SOFT_RAND_MAX 1000
#define MAX_MAT_SIZE 1000
class matrix
{
private:
	int** arr;
	int N;
	void delete_arr();
	void reinit(int n);
public:
	explicit matrix(int n = 3);
	matrix(const matrix& other);
	friend std::ostream& operator<<(std::ostream &os, const matrix& right);
	matrix& operator=(const matrix& other); // copy constructor
	bool operator==(const matrix& right); // compare
	bool operator!=(const matrix& right); // unequal
	matrix operator+(const matrix& right); // add operator
	matrix operator-(const matrix& right); // substract operator
	matrix operator*(const matrix& other);// operator *
	matrix cuda_mult(const matrix &other);//matrix cuda_mult(const matrix& other);
	int* operator[](int x);
	int** get_arr();
	~matrix();
};

int random_number();


