# stdarg, ellipsis and CUDA

Ellipsis are not supported in `__device__` functions, but are a requirement for the C standard library to function correctly.

Libcu implements and supports ellipsis using a unique combiantion of defines and templates

A standard C implementation of ellipsis would be something like `PrintFloats` in the following:
```
#include <stdio.h>
#include <stdarg.h>

void PrintFloats(int n, ...)
{
	va_list va; va_start(va, n);
	printf("Printing floats:");
	for (int i = 0; i < n; i++) {
		double val = va_arg(va, double);
		printf(" [%.2f]", val);
	}
	printf("\n");
	va_end(va);
}
```
And `PrintFloats` would be called with:
```
PrintFloats(3, 3.14159, 2.71828, 1.41421);
```
Refactoring `PrintFloats` by pulling out the `va_list` would result in:
```
void PrintFloats(int n, va_list &va)
{
	printf("Printing floats:");
	for (int i = 0; i < n; i++) {
		double val = va_arg(va, double);
		printf(" [%.2f]", val);
	}
	printf("\n");
}

void PrintFloats(int n, ...)
{
    va_list va; va_start(va, n);
    PrintFloats(n, va);
    va_end(va);
}
```
Using the `libcu` stdarg support for ellipse in `__device__` functions, `PrintFloats` would be implemented as:
```
#include <stdiocu.h>
#include <stdargcu.h>

__device__ void PrintFloats(int n, va_list va)
{
	printf("Printing floats:");
	for (int i = 0; i < n; i++) {
		double val = va_arg(va, double);
		printf(" [%.2f]", val);
	}
	printf("\n");
}
STDARGvoid(PrintFloats(n, va), int n);
```


## STDARG, STDARGvoid 

`STDARG` or `STDARG1-3` is used to implement a ellipse function with a return value, while `STDARGvoid` or `STDARGvoid1-3` is used to implement a ellipse function without a return value.
* type - type of return value
* name - ellipse function name
* name2 - va_list function name
* paramX - parameter symbol name
* paramXCode - parameter code name
```
STDARG({type}, {name}, {name2}(va));
STDARG({type}, {name}, {name2}({param1}, va), {param1Code});
STDARG({type}, {name}, {name2}({param1}, {param2}, va), {param1Code}, {param2Code});
STDARGvoid({name}, {name2}(va));
STDARGvoid({name}, {name2}({param1}, va), {param1Code});
STDARGvoid({name}, {name2}({param1}, {param2}, va), {param1Code}, {param2Code});
```

for simple cases with up to 3 variable parameters the one of the following defines can be used:
* `STDARG` returns {type} with parameters T1 to T3
* `STDARGvoid` returns void with parameters T1 to T3

or for up to 20 variable parameters use the following defines:
* `STDARG1` returns {type} with parameters 01 to 10
* `STDARG2` returns {type} with parameters 11 to 15
* `STDARG3` returns {type} with parameters 16 to 20
* `STDARG1void` returns void with parameters 01 to 10
* `STDARG2void` returns void with parameters 11 to 15
* `STDARG3void` returns void with parameters 16 to 20


example
```
__device__ int methodRet(int cnt, va_list va);
STDARG1(int, methodRet, methodRet(cnt, va), int cnt);
STDARG2(int, methodRet, methodRet(cnt, va), int cnt);
STDARG3(int, methodRet, methodRet(cnt, va), int cnt);
```
or
```
__device__ void methodVoid(int cnt, va_list va);
STDARG1void(methodRet, methodRet(cnt, va), int cnt);
STDARG2void(methodRet, methodRet(cnt, va), int cnt);
```
or
```
__device__ void methodVoid(int cnt, va_list va);
STDARGvoid(methodRet, methodRet(cnt, va), int cnt);
```

## STDARG
The `STDARG` macros and the `va_arg` method will be your primary access methods

`STDARG` does most of the work initilizing state with `va_start`, calling your method, then shutting down state with `va_end`
```
#define STDARG(ret, name, body, ...) \
	__forceinline __device__ ret name(__VA_ARGS__) { _crt_va_list va; _crt_va_start(va); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1> __forceinline __device__ ret name(__VA_ARGS__, T1 arg1) { _crt_va_list1<T1> va; _crt_va_start(va, arg1); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2> __forceinline __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2) { _crt_va_list2<T1,T2> va; _crt_va_start(va, arg1, arg2); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3> __forceinline __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3) { _crt_va_list3<T1,T2,T3> va; _crt_va_start(va, arg1, arg2, arg3); ret r = (body); _crt_va_end(va); return r; }
```

# va_start and state
In stdarg, state is stored using a simple structure defined by templates for each unique variation.

* `_crt_va_list0` is the base for all stdarg state structures
* `b` - is the base field and stores the begining of the structure used for the `va_restart` method
* `i` - is the index field and stores the current location on the structure, this is advanced with the `va_arg` method
* `v1` to `v3` hold the variable state with types `T1` to `T3` respectivly
```
struct _crt_va_list0 { char *b; char *i; };
template <typename T1> struct _crt_va_list1 : _crt_va_list0 { T1 v1; };
template <typename T1, typename T2> struct _crt_va_list2 : _crt_va_list0 { T1 v1; T2 v2; };
template <typename T1, typename T2, typename T3> struct _crt_va_list3 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; };

#define _crt_va_list _crt_va_list0 
#define va_list _crt_va_list0
```

this state is initialized on `va_start` using one of the methods below:
```
static __forceinline __device__ void _crt_va_start(_crt_va_list &va) {
	va.b = va.i = nullptr;
}
template <typename T1>
static __forceinline __device__ void _crt_va_start(_crt_va_list1<T1> &va, T1 arg1) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1;
}
template <typename T1, typename T2>
static __forceinline __device__ void _crt_va_start(_crt_va_list2<T1,T2> &va, T1 arg1, T2 arg2) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2;
}
template <typename T1, typename T2, typename T3>
static __forceinline __device__ void _crt_va_start(_crt_va_list3<T1,T2,T3> &va, T1 arg1, T2 arg2, T3 arg3) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3;
}

#define va_start _crt_va_start
```

after `va_start` the following methods can be used:
* `va_restart` points va_arg back to first parameter
* `va_arg` returns value and advances to next parameter
* `va_end` shutsdown the state
```
#define _crt_va_restart(ap, ...) (ap.i = ap.b);
#define _crt_va_arg(ap, t) (*(t *)((ap.i = (char *)_ROUNDT((unsigned long long)(ap.i + _INTSIZEOF(t)), t)) - _INTSIZEOF(t)))
#define _crt_va_end(ap) (ap.i = nullptr);

#define va_restart _crt_va_restart
#define va_arg _crt_va_arg
#define va_end _crt_va_end
```