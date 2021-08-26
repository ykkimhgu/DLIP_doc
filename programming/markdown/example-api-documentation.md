---
description: Numerical Method example
---

# Example: API documentation

## Reference of Numerical Programming API

`#include "myNM.h"`

## Non-Linear Solver

### newtonRaphson\(\)

Solves the non-linear problem using Newton-Raphson method

```text
double newtonRaphson(double x0, double tol);
```

**Parameters**

* **x0:**  initial value.
* **tol**:  tolerance error

**Example code**

```text
double tol = 0.00001;
double x0 = 3;
double NR_result;
​
NR_result = newtonRaphson(x0, tol);
```

## Linear Solver

### gaussElim\(\)

solves for vector **x** from Ax=b, a linear system problem Using Gauss Elimination

```cpp
void gaussElim(Matrix _A, Matrix _B, Matrix* _U, Matrix* _B_out);
```

**Parameters**

* **A**: Matrix **A** in structure Matrix form. Should be \(nxn\) square.
* **B**: vector **b** in structure Matrix form. Should be \(nx1\)
* **U**: Matrix **U** in structure Matrix form. Should be \(nxn\) square.
* **B\_out**: vector **B\_out** in structure Matrix form. Should be \(nx1\)

**Example code**

```text
Matrix matA = txt2Mat(path, "prob1_matA");
Matrix vecb = txt2Mat(path, "prob1_vecb");
Matrix matU = zeros(matA.rows, matA.cols);
Matrix vecd = zeros(vecb.rows, vecb.cols);
​
gaussElim(matA, vecb, matU, vecd);
```

### inv\(\)

Find the inverse Matrix.

```text
void inv(Matrix _A, Matrix _Ainv);
```

**Parameters**

* **A**: Matrix **A** in structure Matrix form. Should be \(nxn\) square.
* **Ainv**: Matrix **Ainv** in structure Matrix form. Should be \(nxn\) square.

**Example code**

```text
Matrix matA = txt2Mat(path, "prob1_matA");
Matrix matAinv = zeros(matA.rows, matA.cols);
​
inv(matA, matAinv);
```

### 

## Numerical Differentiation

### gradient1D\(\)

Solve for numerical gradient \(dy/dt\) from a 1D-array form.

```text
void gradient1D(double x[], double y[], double dydx[], int m);
```

**Parameters**

* **x\[\]**: input data vector **x** in 1D-array .
* **y\[\]**: input data vector **y** in 1D-array.
* **dydx\[\]**: output vector **dydx** in 1D-array.
* **m**:  length **x** and **y**.

**Example code**

```cpp
double x[21];
    for (int i = 0; i < 21; i++) {
        x[i] = 0.2 * i;
    }
double y[] = { -5.87, -4.23, -2.55, -0.89, 0.67, 2.09, 3.31, 4.31, 5.06, 5.55, 5.78, 5.77, 5.52, 5.08, 4.46, 3.72, 2.88, 2.00, 1.10, 0.23, -0.59 };
double dydx[21];
​
gradient1D(x, y, dydx, 21);
```

## Integration

### integral\(\)

Integral using Simpson 1/3 Method.

```text
double integral(double func(const double _x), double a, double b, int n);
```

**Parameters**

* **func**: Function **func** is defined.
* **a** is starting point of x.
* **b** is ending point of x.
* **n** is the length between **a** and **b**

**Example code**

```text
double I_simpson13 = integral(myFunc, -1, 1, 12);

double myFunc(const double _x) {
	return sqrt(1 - (_x * _x));
}
```

## ODE-IVP

### odeEU\(\)

Solve the 1st-order ODE using Euler's Explicit Method.

```text
void odeEU(double func(const double x, const double y), double y[], double t0, double tf, double h, double y0);
```

**Parameters**

* **func**: Function **func** is defined.
* **y\[\]**: Solution of ODE in structure 1D-array form.
* **t0** is starting point.
* **tf** is ending point.
* **h** is length of step.
* **y0** is initial value of **y\[\]**.

**Example code**

```text
double a = 0;
double b = 0.1;
double h = 0.001;
double y_EU[200] = { 0 };
double v0 = 0;

odeEU(odeFunc_rc, y_EU, a, b, h, v0);

double odeFunc_rc(const double t, const double v) {
	double tau = 1;
	double T = 1 / tau;
	double f = 10;
	double Vm = 1;
	double omega = 2 * PI * f;
	return  -T * v + T * Vm * cos(omega * t);
}
```

-------------------------------------------------------------------------------------------------------

## Class or Header name

### Function Name

```text

```

**Parameters**

* p1
* p2

**Example code**

```text

```

