# Optimized Binary GCD for Modular Inversion

This repository contains documentation and example code that
demonstrates a practical optimization on the implementation of the
binary GCD algorithm, for the purpose of computing modular inverses. The
sample code computes inverses modulo the well-known prime 2^255-19. This
is C code with intrinsics and inline assembly; it requires GCC or Clang,
a 64-bit x86 architecture, and a target CPU that supports the BMI2
opcodes (i.e. Haswell or later, in the line of Intel CPU).

On an Intel Core i5-8259U (Coffee Lake), this code was measured to
perform an inversion in 7490 cycles, which is faster than using Fermat's
little theorem. This code is constant-time, i.e. its execution time and
memory access pattern does not depend on the value to invert (even if
the value is not invertible, in which case the computed "inverse" is
zero).

The implementation of computations modulo 2^255-19 is in
`src/gf25519.c`, with an API in `src/gf25519.h`. All other files are for
test and benchmarking purposes only. Tests also use
[gmplib](https://gmplib.org/) to verify that we compute correct values
(i.e. install the `libgmp-dev` package on your system if you want to
compile and run the tests).
