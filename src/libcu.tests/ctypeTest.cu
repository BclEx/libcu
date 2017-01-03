#include <stdiocu.h>
#include <ctypecu.h>
#include <cuda_runtimecu.h>
#include <assert.h>

static __global__ void ctype_test1()
{
	printf("ctype_test1\n");
	char a0 = toupper('a'); char a0n = toupper('A'); assert(a0 == 'A' || a0n == 'A');
	char a0_ = _toupper('a'); char a0n_ = _toupper('A'); assert(a0_ == 'A' || a0n_ == 'A');
	bool b0 = isupper('a'); bool b0n = isupper('A'); assert(!b0 && b0n);
	bool a1 = isspace('a'); bool a1n = isspace(' '); assert(!a1 && a1n);
	bool a2 = isalnum('a'); bool a2n = isalnum('1'); assert(a2 && a2n);
	bool a3 = isalpha('a'); bool a3n = isalpha('A'); assert(a3 && a3n);
	bool a4 = isdigit('a'); bool a4n = isdigit('1'); assert(!a4 && a4n);
	bool a5 = isxdigit('a'); bool a5n = isxdigit('A'); assert(a5 && a5n);
	char a6 = tolower('a'); char a6n = tolower('A'); assert(a6 == 'a' && a6n == 'a');
	char a6_ = _tolower('a'); char a6n_ = _tolower('A'); assert(a6_ == 'a' && a6n_ == 'a');
	bool b6 = islower('a'); bool b6n = islower('A'); assert(b6 && !b6n);
	//bool a7 = ispoweroftwo(2); bool a7n = ispoweroftwo(3); assert(a7 && !a7n);
	//bool a8 = isalpha2('a'); bool a8n = isalpha2('A'); assert(a8 && a8n);
}

void ctype_()
{
	ctype_test1<<<1, 1>>>();
}
