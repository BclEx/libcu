/*
stdarg.h - defines ANSI-style macros for variable argument functions
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#ifdef __CUDA_ARCH__
#include <stdarg.h>
#elif !defined(_INC_STDARG)
#define _INC_STDARG

//#define COMMA ,
//#define voidSTDARG(name, params1, ...) \
//	__device__ __forceinline void name(__VA_ARGS__) { va_list va; _va_start(va); name##_(params1, va); va_end(va); } \
//	template <typename T1> __device__ __forceinline void name(__VA_ARGS__, T1 arg1) { _va_list1<T1> va; _va_start(va, arg1); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2) { _va_list2<T1,T2> va; _va_start(va, arg1, arg2); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2, typename T3> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3) { _va_list3<T1,T2,T3> va; _va_start(va, arg1, arg2, arg3); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { _va_list4<T1,T2,T3,T4> va; _va_start(va, arg1, arg2, arg3, arg4); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { _va_list5<T1,T2,T3,T4,T5> va; _va_start(va, arg1, arg2, arg3, arg4, arg5); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { _va_list6<T1,T2,T3,T4,T5,T6> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { _va_list7<T1,T2,T3,T4,T5,T6,T7> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { _va_list8<T1,T2,T3,T4,T5,T6,T7,T8> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { _va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { _va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); name##_(params1, va); va_end(va); }
//#define voidSTDARG2(name, params1, ...) \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { _va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { _va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { _va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { _va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); name##_(params1, va); va_end(va); } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { _va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); name##_(params1, va); va_end(va); }


#define STDARG(ret, name, body, ...) \
	__device__ __forceinline void name(__VA_ARGS__) { va_list va; _va_start(va); ret r = body; va_end(va); return r; } \
	template <typename T1> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1) { _va_list1<T1> va; _va_start(va, arg1); ret r = body; va_end(va); return r; } \
	template <typename T1, typename T2> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2) { _va_list2<T1,T2> va; _va_start(va, arg1, arg2); ret r = body; va_end(va); return r; } \
	template <typename T1, typename T2, typename T3> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3) { _va_list3<T1,T2,T3> va; _va_start(va, arg1, arg2, arg3); ret r = body; va_end(va); return r; }


//#define STDARG(ret, name, params1, ...) \
//	__device__ __forceinline void name(__VA_ARGS__) { va_list va; _va_start(va); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1) { _va_list1<T1> va; _va_start(va, arg1); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2) { _va_list2<T1,T2> va; _va_start(va, arg1, arg2); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2, typename T3> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3) { _va_list3<T1,T2,T3> va; _va_start(va, arg1, arg2, arg3); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { _va_list4<T1,T2,T3,T4> va; _va_start(va, arg1, arg2, arg3, arg4); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { _va_list5<T1,T2,T3,T4,T5> va; _va_start(va, arg1, arg2, arg3, arg4, arg5); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { _va_list6<T1,T2,T3,T4,T5,T6> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { _va_list7<T1,T2,T3,T4,T5,T6,T7> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { _va_list8<T1,T2,T3,T4,T5,T6,T7,T8> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { _va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { _va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); ret r = name##_(params1, va); va_end(va); return r; }
//#define STDARG2(ret, name, params1, ...) \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { _va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { _va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { _va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { _va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); ret r = name##_(params1, va); va_end(va); return r; } \
//	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { _va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> va; _va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); ret r = name##_(params1, va); va_end(va); return r; }

struct _va_list0 { char *b; char *i; };
template <typename T1> struct _va_list1 : _va_list0 { T1 v1; };
template <typename T1, typename T2> struct _va_list2 : _va_list0 { T1 v1; T2 v2; };
template <typename T1, typename T2, typename T3> struct _va_list3 : _va_list0 { T1 v1; T2 v2; T3 v3; };
template <typename T1, typename T2, typename T3, typename T4> struct _va_list4 : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; };
template <typename T1, typename T2, typename T3, typename T4, typename T5> struct _va_list5 : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> struct _va_list6 : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> struct _va_list7 : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> struct _va_list8 : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> struct _va_list9 : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> struct _va_listA : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; };
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> struct _va_listB : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> struct _va_listC : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> struct _va_listD : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> struct _va_listE : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> struct _va_listF : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; };
// extended-2
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11> struct _va_list11 : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12> struct _va_list12 : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; T12 v12; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13> struct _va_list13 : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; T12 v12; T13 v13; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14> struct _va_list14 : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; T12 v12; T13 v13; T14 v14; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14, typename T15> struct _va_list15 : _va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; T12 v12; T13 v13; T14 v14; T15 v15; };

#ifndef _INTSIZEOF
#define _INTSIZEOF(n) ((sizeof(n) + sizeof(int) - 1) & ~(sizeof(int) - 1))
#endif
#define va_list _va_list0 
#define va_arg(ap, t) (*(t *)((ap.i = (char *)_ROUNDT(t, (unsigned long long)(ap.i + _INTSIZEOF(t)))) - _INTSIZEOF(t)))
#define va_end(ap) (ap.i = nullptr);

__device__ __forceinline static void _va_restart(va_list &args) {
	args.i = args.b;
}
__device__ __forceinline static void _va_start(va_list &args) {
	args.b = args.i = nullptr;
}
template <typename T1> __device__ __forceinline static void _va_start(_va_list1<T1> &args, T1 arg1) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1;
}
template <typename T1, typename T2> __device__ __forceinline static void _va_start(_va_list2<T1,T2> &args, T1 arg1, T2 arg2) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2;
}
template <typename T1, typename T2, typename T3> __device__ __forceinline static void _va_start(_va_list3<T1,T2,T3> &args, T1 arg1, T2 arg2, T3 arg3) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static void _va_start(_va_list4<T1,T2,T3,T4> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static void _va_start(_va_list5<T1,T2,T3,T4,T5> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static void _va_start(_va_list6<T1,T2,T3,T4,T5,T6> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static void _va_start(_va_list7<T1,T2,T3,T4,T5,T6,T7> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static void _va_start(_va_list8<T1,T2,T3,T4,T5,T6,T7,T8> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static void _va_start(_va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static void _va_start(_va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA;
}
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __device__ __forceinline static void _va_start(_va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __device__ __forceinline static void _va_start(_va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __device__ __forceinline static void _va_start(_va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __device__ __forceinline static void _va_start(_va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __device__ __forceinline static void _va_start(_va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE; args.vF = argF;
}
// extended-2
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11> __device__ __forceinline static void _va_start(_va_list11<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE; args.vF = argF; args.v11 = arg11;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12> __device__ __forceinline static void _va_start(_va_list12<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE; args.vF = argF; args.v11 = arg11; args.v12 = arg12;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13> __device__ __forceinline static void _va_start(_va_list13<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE; args.vF = argF; args.v11 = arg11; args.v12 = arg12; args.v13 = arg13;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14> __device__ __forceinline static void _va_start(_va_list14<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE; args.vF = argF; args.v11 = arg11; args.v12 = arg12; args.v13 = arg13; args.v14 = arg14;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14, typename T15> __device__ __forceinline static void _va_start(_va_list15<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14,T15> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14, T15 arg15) {
	args.b = args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA; args.vB = argB; args.vC = argC; args.vD = argD; args.vE = argE; args.vF = argF; args.v11 = arg11; args.v12 = arg12; args.v13 = arg13; args.v14 = arg14; args.v15 = arg15;
}

#endif  /* _INC_STDARG */