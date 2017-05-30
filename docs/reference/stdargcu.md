---
id: stdargcu
title: stdargcu.h
permalink: stdargcu.html
layout: docs
---

## #include <stdargcu.h>

Also includes:
```
#include <stdarg.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```#define STDARGvoid(name, body, ...)``` | parameters 1 - 3, returns void | #stdarg
```#define STDARG1void(name, body, ...)``` | parameters 1 - 10, returns void | #stdarg1
```#define STDARG2void(name, body, ...)``` | parameters 11 - 15, returns void | #stdarg2
```#define STDARG3void(name, body, ...)``` | parameters 16 - 20, returns void | #stdarg3
```#define STDARG(ret, name, body, ...)``` | parameters 1 - 3, returns "ret" | #stdarg
```#define STDARG1(ret, name, body, ...)``` | parameters 1 - 10, returns "ret" | #stdarg1
```#define STDARG2(ret, name, body, ...)``` | parameters 11 - 15, returns "ret" | #stdarg2
```#define STDARG3(ret, name, body, ...)``` | parameters 16 - 20, returns "ret" | #stdarg3
```#define va_list``` | the variable argument list (va_list)
```#define va_start``` | start a va_list
```#define va_restart``` | restart a va_list
```#define va_arg``` | next argument in va_list
```#define va_end``` | close a va_list
