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

//#pragma once
#ifndef _STDARGCU_H
#define _STDARGCU_H

#include <stdarg.h>
#if defined(__CUDA_ARCH__)

#define STDARGvoid(name, body, ...) \
	__forceinline__ __device__ void name(__VA_ARGS__) { _crt_va_list va; _crt_va_start(va); (body); _crt_va_end(va); } \
	template <typename T1> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1) { _crt_va_list1<T1> va; _crt_va_start(va, arg1); (body); _crt_va_end(va); } \
	template <typename T1, typename T2> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2) { _crt_va_list2<T1,T2> va; _crt_va_start(va, arg1, arg2); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3) { _crt_va_list3<T1,T2,T3> va; _crt_va_start(va, arg1, arg2, arg3); (body); _crt_va_end(va); }
#define STDARG1void(name, body, ...) \
	__forceinline__ __device__ void name(__VA_ARGS__) { _crt_va_list va; _crt_va_start(va); (body); _crt_va_end(va); } \
	template <typename T1> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1) { _crt_va_list1<T1> va; _crt_va_start(va, arg1); (body); _crt_va_end(va); } \
	template <typename T1, typename T2> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2) { _crt_va_list2<T1,T2> va; _crt_va_start(va, arg1, arg2); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3) { _crt_va_list3<T1,T2,T3> va; _crt_va_start(va, arg1, arg2, arg3); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { _crt_va_list4<T1,T2,T3,T4> va; _crt_va_start(va, arg1, arg2, arg3, arg4); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { _crt_va_list5<T1,T2,T3,T4,T5> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { _crt_va_list6<T1,T2,T3,T4,T5,T6> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { _crt_va_list7<T1,T2,T3,T4,T5,T6,T7> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { _crt_va_list8<T1,T2,T3,T4,T5,T6,T7,T8> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { _crt_va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { _crt_va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); (body); _crt_va_end(va); }
// extended
#define STDARG2void(name, body, ...) \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { _crt_va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { _crt_va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { _crt_va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { _crt_va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { _crt_va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); (body); _crt_va_end(va); }
// extended-2
#define STDARG3void(name, body, ...) \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11) { _crt_va_list11<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12) { _crt_va_list12<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13) { _crt_va_list13<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12, arg13); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14) { _crt_va_list14<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12, arg13, arg14); (body); _crt_va_end(va); } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14, typename T15> __forceinline__ __device__ void name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14, T15 arg15) { _crt_va_list15<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14,T15> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12, arg13, arg14, arg15); (body); _crt_va_end(va); } \

#define STDARG(ret, name, body, ...) \
	__forceinline__ __device__ ret name(__VA_ARGS__) { _crt_va_list va; _crt_va_start(va); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1) { _crt_va_list1<T1> va; _crt_va_start(va, arg1); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2) { _crt_va_list2<T1,T2> va; _crt_va_start(va, arg1, arg2); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3) { _crt_va_list3<T1,T2,T3> va; _crt_va_start(va, arg1, arg2, arg3); ret r = (body); _crt_va_end(va); return r; }
#define STDARG1(ret, name, body, ...) \
	__forceinline__ __device__ ret name(__VA_ARGS__) { _crt_va_list va; _crt_va_start(va); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1) { _crt_va_list1<T1> va; _crt_va_start(va, arg1); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2) { _crt_va_list2<T1,T2> va; _crt_va_start(va, arg1, arg2); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3) { _crt_va_list3<T1,T2,T3> va; _crt_va_start(va, arg1, arg2, arg3); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { _crt_va_list4<T1,T2,T3,T4> va; _crt_va_start(va, arg1, arg2, arg3, arg4); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { _crt_va_list5<T1,T2,T3,T4,T5> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { _crt_va_list6<T1,T2,T3,T4,T5,T6> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { _crt_va_list7<T1,T2,T3,T4,T5,T6,T7> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { _crt_va_list8<T1,T2,T3,T4,T5,T6,T7,T8> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { _crt_va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { _crt_va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); ret r = (body); _crt_va_end(va); return r; }
// extended
#define STDARG2(ret, name, body, ...) \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) { _crt_va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) { _crt_va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) { _crt_va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) { _crt_va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) { _crt_va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF); ret r = (body); _crt_va_end(va); return r; }
// extended-2
#define STDARG3(ret, name, body, ...) \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11) { _crt_va_list11<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12) { _crt_va_list12<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13) { _crt_va_list13<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12, arg13); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14) { _crt_va_list14<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12, arg13, arg14); ret r = (body); _crt_va_end(va); return r; } \
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14, typename T15> __forceinline__ __device__ ret name(__VA_ARGS__, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14, T15 arg15) { _crt_va_list15<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14,T15> va; _crt_va_start(va, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA, argB, argC, argD, argE, argF, arg11, arg12, arg13, arg14, arg15); ret r = (body); _crt_va_end(va); return r; } \

struct _crt_va_list0 { char *b; char *i; };
template <typename T1> struct _crt_va_list1 : _crt_va_list0 { T1 v1; };
template <typename T1, typename T2> struct _crt_va_list2 : _crt_va_list0 { T1 v1; T2 v2; };
template <typename T1, typename T2, typename T3> struct _crt_va_list3 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; };
template <typename T1, typename T2, typename T3, typename T4> struct _crt_va_list4 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; };
template <typename T1, typename T2, typename T3, typename T4, typename T5> struct _crt_va_list5 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> struct _crt_va_list6 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> struct _crt_va_list7 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> struct _crt_va_list8 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> struct _crt_va_list9 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> struct _crt_va_listA : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; };
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> struct _crt_va_listB : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> struct _crt_va_listC : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> struct _crt_va_listD : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> struct _crt_va_listE : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> struct _crt_va_listF : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; };
// extended-2
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11> struct _crt_va_list11 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12> struct _crt_va_list12 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; T12 v12; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13> struct _crt_va_list13 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; T12 v12; T13 v13; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14> struct _crt_va_list14 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; T12 v12; T13 v13; T14 v14; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14, typename T15> struct _crt_va_list15 : _crt_va_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; TB vB; TC vC; TD vD; TE vE; TF vF; T11 v11; T12 v12; T13 v13; T14 v14; T15 v15; };

#undef _INTSIZEOF
#undef _crt_va_start
#undef _crt_va_arg
#undef _crt_va_end

#ifndef _INTSIZEOF
#define _INTSIZEOF(n) ((sizeof(n) + sizeof(int) - 1) & ~(sizeof(int) - 1))
#endif
#define _crt_va_list _crt_va_list0 
#define _crt_va_restart(ap, ...) (ap.i = ap.b);
#ifndef _WIN64
#define _crt_va_arg(ap, t) (*(t *)((ap.i = (char *)_ROUNDT((unsigned long)(ap.i + _INTSIZEOF(t)), t)) - _INTSIZEOF(t)))
#else
#define _crt_va_arg(ap, t) (*(t *)((ap.i = (char *)_ROUNDT((unsigned long long)(ap.i + _INTSIZEOF(t)), t)) - _INTSIZEOF(t)))
#endif
#define _crt_va_end(ap) (ap.i = nullptr);

static __forceinline__ __device__ void _crt_va_start(_crt_va_list &va) {
	va.b = va.i = nullptr;
}
template <typename T1> static __forceinline__ __device__ void _crt_va_start(_crt_va_list1<T1> &va, T1 arg1) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1;
}
template <typename T1, typename T2> static __forceinline__ __device__ void _crt_va_start(_crt_va_list2<T1,T2> &va, T1 arg1, T2 arg2) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2;
}
template <typename T1, typename T2, typename T3> static __forceinline__ __device__ void _crt_va_start(_crt_va_list3<T1,T2,T3> &va, T1 arg1, T2 arg2, T3 arg3) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3;
}
template <typename T1, typename T2, typename T3, typename T4> static __forceinline__ __device__ void _crt_va_start(_crt_va_list4<T1,T2,T3,T4> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> static __forceinline__ __device__ void _crt_va_start(_crt_va_list5<T1,T2,T3,T4,T5> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> static __forceinline__ __device__ void _crt_va_start(_crt_va_list6<T1,T2,T3,T4,T5,T6> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> static __forceinline__ __device__ void _crt_va_start(_crt_va_list7<T1,T2,T3,T4,T5,T6,T7> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> static __forceinline__ __device__ void _crt_va_start(_crt_va_list8<T1,T2,T3,T4,T5,T6,T7,T8> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> static __forceinline__ __device__ void _crt_va_start(_crt_va_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8; va.v9 = arg9;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> static __forceinline__ __device__ void _crt_va_start(_crt_va_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8; va.v9 = arg9; va.vA = argA;
}
// extended
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB> static __forceinline__ __device__ void _crt_va_start(_crt_va_listB<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8; va.v9 = arg9; va.vA = argA; va.vB = argB;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC> static __forceinline__ __device__ void _crt_va_start(_crt_va_listC<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8; va.v9 = arg9; va.vA = argA; va.vB = argB; va.vC = argC;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD> static __forceinline__ __device__ void _crt_va_start(_crt_va_listD<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8; va.v9 = arg9; va.vA = argA; va.vB = argB; va.vC = argC; va.vD = argD;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE> static __forceinline__ __device__ void _crt_va_start(_crt_va_listE<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8; va.v9 = arg9; va.vA = argA; va.vB = argB; va.vC = argC; va.vD = argD; va.vE = argE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF> static __forceinline__ __device__ void _crt_va_start(_crt_va_listF<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8; va.v9 = arg9; va.vA = argA; va.vB = argB; va.vC = argC; va.vD = argD; va.vE = argE; va.vF = argF;
}
// extended-2
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11> static __forceinline__ __device__ void _crt_va_start(_crt_va_list11<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8; va.v9 = arg9; va.vA = argA; va.vB = argB; va.vC = argC; va.vD = argD; va.vE = argE; va.vF = argF; va.v11 = arg11;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12> static __forceinline__ __device__ void _crt_va_start(_crt_va_list12<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8; va.v9 = arg9; va.vA = argA; va.vB = argB; va.vC = argC; va.vD = argD; va.vE = argE; va.vF = argF; va.v11 = arg11; va.v12 = arg12;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13> static __forceinline__ __device__ void _crt_va_start(_crt_va_list13<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8; va.v9 = arg9; va.vA = argA; va.vB = argB; va.vC = argC; va.vD = argD; va.vE = argE; va.vF = argF; va.v11 = arg11; va.v12 = arg12; va.v13 = arg13;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14> static __forceinline__ __device__ void _crt_va_start(_crt_va_list14<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8; va.v9 = arg9; va.vA = argA; va.vB = argB; va.vC = argC; va.vD = argD; va.vE = argE; va.vF = argF; va.v11 = arg11; va.v12 = arg12; va.v13 = arg13; va.v14 = arg14;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA, typename TB, typename TC, typename TD, typename TE, typename TF, typename T11, typename T12, typename T13, typename T14, typename T15> static __forceinline__ __device__ void _crt_va_start(_crt_va_list15<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA,TB,TC,TD,TE,TF,T11,T12,T13,T14,T15> &va, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA, TB argB, TC argC, TD argD, TE argE, TF argF, T11 arg11, T12 arg12, T13 arg13, T14 arg14, T15 arg15) {
	va.b = va.i = (char *)&va.v1; va.v1 = arg1; va.v2 = arg2; va.v3 = arg3; va.v4 = arg4; va.v5 = arg5; va.v6 = arg6; va.v7 = arg7; va.v8 = arg8; va.v9 = arg9; va.vA = argA; va.vB = argB; va.vC = argC; va.vD = argD; va.vE = argE; va.vF = argF; va.v11 = arg11; va.v12 = arg12; va.v13 = arg13; va.v14 = arg14; va.v15 = arg15;
}

//#define _VA_LIST_DEFINED
#define va_list _crt_va_list0
#define va_list1 _crt_va_list1
#define va_list2 _crt_va_list2
#define va_list3 _crt_va_list3
#define va_list4 _crt_va_list4
#define va_list5 _crt_va_list5
#define va_list6 _crt_va_list6
#define va_list7 _crt_va_list7
#define va_list8 _crt_va_list8
#define va_list9 _crt_va_list9
#define va_listA _crt_va_listA
// extended
#define va_listB _crt_va_listB
#define va_listC _crt_va_listC
#define va_listD _crt_va_listD
#define va_listE _crt_va_listE
#define va_listF _crt_va_listF
// extended-2
#define va_list11 _crt_va_list11
#define va_list12 _crt_va_list12
#define va_list13 _crt_va_list13
#define va_list14 _crt_va_list14
#define va_list15 _crt_va_list15

#else
#define STDARGvoid(name, body, ...) __forceinline__ void name(...) { }
#define STDARG1void(name, body, ...) __forceinline__ void name(...) { }
#define STDARG2void(name, body, ...)
#define STDARG3void(name, body, ...)
#define STDARG(ret, name, body, ...) __forceinline__ ret name(...) { return (ret)0; }
#define STDARG1(ret, name, body, ...) __forceinline__ ret name(...) { return (ret)0; }
#define STDARG2(ret, name, body, ...)
#define STDARG3(ret, name, body, ...)
#define _crt_va_restart _crt_va_start
#endif  /* __CUDA_ARCH__ */

#undef va_start
#undef va_arg
#undef va_end
#define va_start _crt_va_start
#define va_restart _crt_va_restart
#define va_arg _crt_va_arg
#define va_end _crt_va_end

#endif  /* _STDARGCU_H */
