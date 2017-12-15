#include "matrix.h"

matrix::matrix(int n)
{
	reinit(n);
}

matrix::matrix(const matrix &other) // copy ctor
{
	reinit(other.N);
	int i, j;
	for (i = 0; i<this->N; i++)
	{
		for (j = 0; j<this->N; j++)
		{
			this->arr[i][j] = other.arr[i][j];
		}
	}
}

matrix &matrix::operator=(const matrix &other)
{
	if (&other == this)
		return *this;
	int i, j;
	if (this->N != other.N)
	{
		this->delete_arr();
		this->reinit(other.N);
	}
	for (i = 0; i<this->N; i++)
	{
		for (j = 0; j<this->N; j++)
		{
			this->arr[i][j] = other.arr[i][j];
		}
	}

	return *this;
}

void matrix::delete_arr()
{
	int i;
	for (i = 0; i<this->N; i++)
		free(this->arr[i]);
	free(this->arr);
}

void matrix::reinit(int n)
{
	this->N = n;
	this->arr = static_cast<int **>(calloc(n, sizeof(int*))); // allocate master array
	int i, j;
	for (i = 0; i<this->N; i++) // allocate inner array
		this->arr[i] = static_cast<int *>(calloc(n, sizeof(int)));

	for (i = 0; i<this->N; i++)
	{
		for (j = 0; j<this->N; j++)
		{
			this->arr[i][j] = 0;
		}
	}
}

bool matrix::operator==(const matrix &right)
{
	int i, j;
	if (this->N != right.N)
		return false;
	for (i = 0; i<this->N; i++)
	{
		for (j = 0; j<this->N; j++)
			if (this->arr[i][j] != right.arr[i][j])
				return false;
	}
	return true;
}

bool matrix::operator!=(const matrix &right)
{
	return !(this->operator==(right));
}



std::ostream& operator<<(std::ostream &os, const matrix& right)
{
	int i, j;
	for (i = 0; i<right.N; i++)
	{
		for (j = 0; j<right.N; j++)
		{
			if (j != 0)
				os << " ";
			os << right.arr[i][j];
		}
		os << std::endl;
	}
	return os;
}

matrix matrix::operator+(const matrix &right)
{
	if (this->N != right.N)
		return matrix();
	matrix matrix1(this->N);
	int i, j;
	for (i = 0; i<this->N; i++)
	{
		for (j = 0; j<this->N; j++)
			matrix1.arr[i][j] = this->arr[i][j] + right.arr[i][j];
	}
	return matrix1;
}

matrix matrix::operator-(const matrix &right)
{
	if (this->N != right.N)
		return matrix();
	matrix matrix1(this->N);
	int i, j;
	for (i = 0; i<this->N; i++)
	{
		for (j = 0; j<this->N; j++)
			matrix1.arr[i][j] = this->arr[i][j] - right.arr[i][j];
	}
	return matrix1;
}

matrix matrix::operator*(const matrix& other)
{
	if (this->N != other.N)
		return matrix();
	matrix mult1(this->N);

	int trds = (omp_get_num_procs() > this->N ) ? this->N : omp_get_num_procs();

	int i;
	#pragma omp parallel for num_threads(trds)
	for (i = 0; i<this->N; i++)
	{
		for (register int j = 0; j<this->N; j++)
		{
			for (register int k = 0; k<this->N; k++)
			{
				mult1.arr[i][j] += this->arr[i][k] * other.arr[k][j];
			}
		}
	}
	return mult1;
}

int* matrix::operator[](int x)
{
	return (x<this->N) ? this->arr[x] : nullptr;
}

int ** matrix::get_arr()
{
	return this->arr;
}

matrix::~matrix()
{
	delete_arr();
}

int random_number()
{
	return (rand() % SOFT_RAND_MAX) + 1;
}







