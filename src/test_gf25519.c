#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * We try to bind to a specific CPU so that measures are easier to
 * reproduce.
 */
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>

/*
 * We use __rdtsc() to access the cycle counter.
 *
 * Ideally we should use __rdpmc(0x40000001) to get the actual cycle
 * counter (independently of frequency scaling), but this requires access
 * to have been enabled:
 *    echo 2 > /sys/bus/event_source/devices/cpu/rdpmc
 * This command requires root privileges, but once it is done, normal
 * user processes can use __rdpmc() to read the cycle counter (the
 * setting is reset at boot time).
 * (If that file contains 1 instead of 2, then we may read the counter
 * as well, but only if we first enable the performance events, which
 * means more code.)
 *
 * For the kind of things we do, though, there should be no practical
 * difference between TSC and the raw cycle counter as long as frequency
 * scaling does not happen after some warmup. In particular, TurboBoost
 * should be disabled, which is usually done in the BIOS screen (if you
 * are testing this on a VM, you might be out of luck).
 *
 * SMT ("HyperThreading") should also be disabled. This can be done
 * (with root privileges) through the following command:
 *    echo off > /sys/devices/system/cpu/smt/control
 */
#include <immintrin.h>

/*
 * gmplib is used in tests to verify that the functions compute things
 * correctly.
 */
#include <gmp.h>

/*
 * A custom SHA-3 / SHAKE implementation is used for pseudorandom (but
 * reproducible) generation of test values.
 */
#include "sha3.h"

#include "gf25519.h"

static void
print_gf(const char *name, const gf *x)
{
	gf t;

	t = *x;
	printf("%s = 0x%016llX%016llX%016llX%016llX\n",
		name,
		(unsigned long long)t.v3, (unsigned long long)t.v2,
		(unsigned long long)t.v1, (unsigned long long)t.v0);
}

static void
rand_init(shake_context *rng, const char *seed)
{
	shake_init(rng, 256);
	shake_inject(rng, seed, strlen(seed));
	shake_flip(rng);
}

static uint64_t
rand_u64(shake_context *rng)
{
	uint8_t tmp[8];
	uint64_t r;
	int i;

	shake_extract(rng, tmp, sizeof tmp);
	r = 0;
	for (i = 0; i < 8; i ++) {
		r = (r << 8) | tmp[i];
	}
	return r;
}

static void
rand_gf(shake_context *rng, gf *d)
{
	d->v0 = rand_u64(rng);
	d->v1 = rand_u64(rng);
	d->v2 = rand_u64(rng);
	d->v3 = rand_u64(rng);
}

static void
gf_to_mpz(mpz_t r, const gf *a)
{
	mpz_set_ui(r, (uint32_t)(a->v3 >> 32));
	mpz_mul_2exp(r, r, 32);
	mpz_add_ui(r, r, (uint32_t)a->v3);

	mpz_mul_2exp(r, r, 32);
	mpz_add_ui(r, r, (uint32_t)(a->v2 >> 32));
	mpz_mul_2exp(r, r, 32);
	mpz_add_ui(r, r, (uint32_t)a->v2);

	mpz_mul_2exp(r, r, 32);
	mpz_add_ui(r, r, (uint32_t)(a->v1 >> 32));
	mpz_mul_2exp(r, r, 32);
	mpz_add_ui(r, r, (uint32_t)a->v1);

	mpz_mul_2exp(r, r, 32);
	mpz_add_ui(r, r, (uint32_t)(a->v0 >> 32));
	mpz_mul_2exp(r, r, 32);
	mpz_add_ui(r, r, (uint32_t)a->v0);
}

void
test_add(void)
{
	shake_context rng;
	gf a, b, c;
	mpz_t za, zb, zc, zt, zq;
	int i;

	printf("Test add: ");
	fflush(stdout);

	mpz_init(za);
	mpz_init(zb);
	mpz_init(zc);
	mpz_init(zt);
	mpz_init(zq);
	mpz_set_ui(zq, 1);
	mpz_mul_2exp(zq, zq, 255);
	mpz_sub_ui(zq, zq, 19);

	rand_init(&rng, "test add");

	for (i = 0; i < 100000; i ++) {
		rand_gf(&rng, &a);
		rand_gf(&rng, &b);

		gf_add(&c, &a, &b);
		gf_to_mpz(za, &a);
		gf_to_mpz(zb, &b);
		mpz_add(zt, za, zb);
		gf_to_mpz(zc, &c);
		mpz_sub(zt, zt, zc);
		mpz_mod(zt, zt, zq);
		if (mpz_sgn(zt) != 0) {
			fprintf(stderr, "ERR: ADD:\n");
			print_gf("a", &a);
			print_gf("b", &b);
			print_gf("c", &c);
			exit(EXIT_FAILURE);
		}

		gf_sub(&c, &a, &b);
		gf_to_mpz(za, &a);
		gf_to_mpz(zb, &b);
		mpz_sub(zt, za, zb);
		gf_to_mpz(zc, &c);
		mpz_sub(zt, zt, zc);
		mpz_mod(zt, zt, zq);
		if (mpz_sgn(zt) != 0) {
			fprintf(stderr, "ERR: SUB:\n");
			print_gf("a", &a);
			print_gf("b", &b);
			print_gf("c", &c);
			exit(EXIT_FAILURE);
		}

		if (i % 4000 == 0) {
			printf(".");
			fflush(stdout);
		}
	}

	mpz_clear(za);
	mpz_clear(zb);
	mpz_clear(zc);
	mpz_clear(zt);
	mpz_clear(zq);

	printf(" done.\n");
	fflush(stdout);
}

void
test_mul(void)
{
	shake_context rng;
	gf a, b, c;
	mpz_t za, zb, zc, zt, zq;
	int i;

	static const gf GF_Q = {
		(unsigned long long)-19,
		(unsigned long long)-1,
		(unsigned long long)-1,
		(unsigned long long)-1 >> 1
	};

	printf("Test mul: ");
	fflush(stdout);

	mpz_init(za);
	mpz_init(zb);
	mpz_init(zc);
	mpz_init(zt);
	mpz_init(zq);
	mpz_set_ui(zq, 1);
	mpz_mul_2exp(zq, zq, 255);
	mpz_sub_ui(zq, zq, 19);

	rand_init(&rng, "test mul");

	for (i = 0; i < 100000; i ++) {
		uint8_t tmp[32];
		int j;

		rand_gf(&rng, &a);
		rand_gf(&rng, &b);
		gf_mul(&c, &a, &b);
		if (gf_iszero(&a) || gf_iszero(&b)) {
			fprintf(stderr, "ERR: gf_iszero()\n");
			exit(EXIT_FAILURE);
		}

		gf_to_mpz(za, &a);
		gf_to_mpz(zb, &b);
		mpz_mul(zt, za, zb);
		gf_to_mpz(zc, &c);
		mpz_sub(zt, zt, zc);
		mpz_mod(zt, zt, zq);
		if (mpz_sgn(zt) != 0) {
			fprintf(stderr, "ERR: MUL:\n");
			gmp_printf("a = %Zd\n", za);
			gmp_printf("b = %Zd\n", zb);
			gmp_printf("c = %Zd\n", zc);
			print_gf("a", &a);
			print_gf("b", &b);
			print_gf("c", &c);
			exit(EXIT_FAILURE);
		}

		memset(&c, 0, sizeof c);
		gf_sqr(&c, &a);
		mpz_mul(zt, za, za);
		gf_to_mpz(zc, &c);
		mpz_sub(zt, zt, zc);
		mpz_mod(zt, zt, zq);
		if (mpz_sgn(zt) != 0) {
			fprintf(stderr, "ERR: SQR:\n");
			gmp_printf("a = %Zd\n", za);
			gmp_printf("c = %Zd\n", zc);
			print_gf("a", &a);
			print_gf("c", &c);
			exit(EXIT_FAILURE);
		}

		for (j = 1; j <= 4; j ++) {
			int k;

			memset(&c, 0, sizeof c);
			gf_sqr_x(&c, &a, j);
			mpz_set(zt, za);
			for (k = 0; k < j; k ++) {
				mpz_mul(zt, zt, zt);
				mpz_mod(zt, zt, zq);
			}
			gf_to_mpz(zc, &c);
			mpz_sub(zt, zt, zc);
			mpz_mod(zt, zt, zq);
			if (mpz_sgn(zt) != 0) {
				fprintf(stderr, "ERR: SQR_X(%d):\n", j);
				gmp_printf("a = %Zd\n", za);
				gmp_printf("c = %Zd\n", zc);
				print_gf("a", &a);
				print_gf("c", &c);
				exit(EXIT_FAILURE);
			}
		}

		gf_encode(tmp, &b);
		gf_decode(&c, tmp);
		gf_to_mpz(za, &c);
		gf_to_mpz(zb, &b);
		mpz_mod(zt, zb, zq);
		if (mpz_cmp(za, zq) >= 0 || mpz_cmp(za, zt) != 0) {
			fprintf(stderr, "ERR: ENCODE/DECODE:\n");
			print_gf("b", &b);
			print_gf("c", &c);
			exit(EXIT_FAILURE);
		}

		gf_mul(&c, &a, &GF_Q);
		if (!gf_iszero(&c)) {
			fprintf(stderr, "ERR: MUL BY ZERO\n");
			exit(EXIT_FAILURE);
		}

		gf_add(&b, &GF_Q, &GF_Q);
		gf_mul(&c, &a, &b);
		if (!gf_iszero(&c)) {
			fprintf(stderr, "ERR: MUL BY ZERO (2)\n");
			exit(EXIT_FAILURE);
		}

		if (i % 4000 == 0) {
			printf(".");
			fflush(stdout);
		}
	}

	mpz_clear(za);
	mpz_clear(zb);
	mpz_clear(zc);
	mpz_clear(zt);
	mpz_clear(zq);

	printf(" done.\n");
	fflush(stdout);
}

void
test_inv(void)
{
	shake_context rng;
	gf a, b, c;
	mpz_t za, zb;
	int i;

	printf("Test inv: ");
	fflush(stdout);

	mpz_init(za);
	mpz_init(zb);

	rand_init(&rng, "test inv");

	for (i = 0; i < 10000; i ++) {
		uint8_t tmp[32];
		static const uint8_t ONE[] = {
			1, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0
		};

		if (i == 0) {
			a.v0 = 0;
			a.v1 = 0;
			a.v2 = 0;
			a.v3 = (uint64_t)1 << 62;
		} else {
			rand_gf(&rng, &a);
		}
		gf_inv(&b, &a);
		gf_mul(&c, &a, &b);
		gf_encode(tmp, &c);
		if (memcmp(tmp, ONE, sizeof ONE) != 0) {
			fprintf(stderr, "ERR: INV:\n");
			gf_to_mpz(za, &a);
			gf_to_mpz(zb, &b);
			gmp_printf("a = %Zd\n", za);
			gmp_printf("b = %Zd\n", zb);
			print_gf("a", &a);
			print_gf("b", &b);
			print_gf("c", &c);
			exit(EXIT_FAILURE);
		}

#if defined GF_INV_FLT && GF_INV_FLT
		memset(&b, 0, sizeof b);
		gf_inv_FLT(&b, &a);
		gf_mul(&c, &a, &b);
		gf_encode(tmp, &c);
		if (memcmp(tmp, ONE, sizeof ONE) != 0) {
			fprintf(stderr, "ERR: INV(FLT):\n");
			gf_to_mpz(za, &a);
			gf_to_mpz(zb, &b);
			gmp_printf("a = %Zd\n", za);
			gmp_printf("b = %Zd\n", zb);
			print_gf("a", &a);
			print_gf("b", &b);
			print_gf("c", &c);
			exit(EXIT_FAILURE);
		}
#endif

		if (i % 400 == 0) {
			printf(".");
			fflush(stdout);
		}
	}

	mpz_clear(za);
	mpz_clear(zb);

	printf(" done.\n");
	fflush(stdout);
}

static int
cmp_u64(const void *p1, const void *p2)
{
	uint64_t v1, v2;

	v1 = *(const uint64_t *)p1;
	v2 = *(const uint64_t *)p2;
	if (v1 < v2) {
		return -1;
	} else if (v1 == v2) {
		return 0;
	} else {
		return 1;
	}
}

/*
 * Read the cycle counter. The 'lfence' call should guarantee enough
 * serialization without adding too much overhead (contrary to what,
 * say, 'cpuid' would do).
 */
static inline uint64_t
core_cycles(void)
{
	/*
	 * GCC seems not to have the __rdtsc() intrinsic.
	 */
#if defined __GNUC__ && !defined __clang__
	uint32_t hi, lo;

	_mm_lfence();
	__asm__ __volatile__ ("rdtsc" : "=d" (hi), "=a" (lo) : : );
	return ((uint64_t)hi << 32) | (uint64_t)lo;
#else
	_mm_lfence();
	return __rdtsc();
#endif
}

static void
speed_mul(void)
{
	size_t u;
	uint64_t tt[1000];
	gf a, b, c;

#define NUM   ((sizeof tt) / (sizeof tt[0]))

	memset(&a, 'T', sizeof a);
	memset(&b, 'P', sizeof b);
	for (u = 0; u < 2 * NUM; u ++) {
		uint64_t begin, end;
		int i;

		begin = core_cycles();
		for (i = 0; i < 1000; i ++) {
			gf_mul(&c, &a, &b);
			gf_mul(&a, &b, &c);
			gf_mul(&b, &c, &a);
			gf_mul(&c, &a, &b);
			gf_mul(&a, &b, &c);
			gf_mul(&b, &c, &a);
		}
		end = core_cycles();
		if (u >= NUM) {
			tt[u - NUM] = end - begin;
		}
	}
	qsort(tt, (sizeof tt) / sizeof(tt[0]), sizeof tt[0], &cmp_u64);
	printf("multiplication:         %9.2f (%.2f .. %.2f)\n",
		(double)tt[NUM / 2] / 6000.0,
		(double)tt[NUM / 10] / 6000.0,
		(double)tt[(9 * NUM) / 10] / 6000.0);
	fflush(stdout);

#undef NUM
}

static void
speed_sqr(void)
{
	size_t u;
	uint64_t tt[1000];
	gf a, b, c;

#define NUM   ((sizeof tt) / (sizeof tt[0]))

	memset(&a, 'T', sizeof a);
	for (u = 0; u < 2 * NUM; u ++) {
		uint64_t begin, end;
		int i;

		begin = core_cycles();
		for (i = 0; i < 1000; i ++) {
			gf_sqr(&b, &a);
			gf_sqr(&c, &b);
			gf_sqr(&a, &c);
			gf_sqr(&b, &a);
			gf_sqr(&c, &b);
			gf_sqr(&a, &c);
		}
		end = core_cycles();
		if (u >= NUM) {
			tt[u - NUM] = end - begin;
		}
	}
	qsort(tt, (sizeof tt) / sizeof(tt[0]), sizeof tt[0], &cmp_u64);
	printf("squaring:               %9.2f (%.2f .. %.2f)\n",
		(double)tt[NUM / 2] / 6000.0,
		(double)tt[NUM / 10] / 6000.0,
		(double)tt[(9 * NUM) / 10] / 6000.0);
	fflush(stdout);

#undef NUM
}

static void
speed_inv(void)
{
	size_t u;
	uint64_t tt[1000];
	gf a, b;

#define NUM   ((sizeof tt) / (sizeof tt[0]))

	memset(&a, 'T', sizeof a);
	memset(&b, 'P', sizeof b);
	for (u = 0; u < 2 * NUM; u ++) {
		uint64_t begin, end;
		int i;

		begin = core_cycles();
		for (i = 0; i < 25; i ++) {
			gf_inv(&b, &a);
			gf_inv(&a, &b);
			gf_inv(&b, &a);
			gf_inv(&a, &b);
		}
		end = core_cycles();
		if (u >= NUM) {
			tt[u - NUM] = end - begin;
		}
	}
	qsort(tt, (sizeof tt) / sizeof(tt[0]), sizeof tt[0], &cmp_u64);
	printf("inversion (binary GCD): %9.2f (%.2f .. %.2f)\n",
		(double)tt[NUM / 2] / 100.0,
		(double)tt[NUM / 10] / 100.0,
		(double)tt[(9 * NUM) / 10] / 100.0);
	fflush(stdout);

#undef NUM
}

#if defined GF_INV_FLT && GF_INV_FLT
static void
speed_inv_FLT(void)
{
	size_t u;
	uint64_t tt[1000];
	gf a, b;

#define NUM   ((sizeof tt) / (sizeof tt[0]))

	memset(&a, 'T', sizeof a);
	memset(&b, 'P', sizeof b);
	for (u = 0; u < 2 * NUM; u ++) {
		uint64_t begin, end;
		int i;

		begin = core_cycles();
		for (i = 0; i < 25; i ++) {
			gf_inv_FLT(&b, &a);
			gf_inv_FLT(&a, &b);
			gf_inv_FLT(&b, &a);
			gf_inv_FLT(&a, &b);
		}
		end = core_cycles();
		if (u >= NUM) {
			tt[u - NUM] = end - begin;
		}
	}
	qsort(tt, (sizeof tt) / sizeof(tt[0]), sizeof tt[0], &cmp_u64);
	printf("inversion (FLT):        %9.2f (%.2f .. %.2f)\n",
		(double)tt[NUM / 2] / 100.0,
		(double)tt[NUM / 10] / 100.0,
		(double)tt[(9 * NUM) / 10] / 100.0);
	fflush(stdout);

#undef NUM
}
#endif

int
main(int argc, char *argv[])
{
	/*
	 * If given an explicit argument, then we use it as a CPU/core
	 * identifier to bind ourselves.
	 */
	if (argc >= 2) {
		cpu_set_t cs;
		int cpu;

		cpu = atoi(argv[1]);
		printf("(binding to CPU %d)\n", cpu);
		CPU_ZERO(&cs);
		CPU_SET(cpu, &cs);
		if (sched_setaffinity(getpid(), sizeof cs, &cs) != 0) {
			perror("sched_setaffinity");
		}
	}

	test_add();
	test_mul();
	test_inv();

	printf("Speed benchmark (median, low 10%%, high 10%%):\n");
	fflush(stdout);
	speed_mul();
	speed_sqr();
	speed_inv();
#if defined GF_INV_FLT && GF_INV_FLT
	speed_inv_FLT();
#endif
	return 0;
}
