#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <immintrin.h>

#include "gf25519.h"

/*
 * We work in field Z_p with p = 2^255-MQ. The code below can be
 * used to work modulo any odd integer with that format, provided that
 * the macros below are adapted accordingly, and subject to the
 * following restrictions:
 *
 *    MQ must be odd and in the 1..2147483647 range
 *    GF_INVT508 must contain 1/2^508 mod p (64-bit limbs, little-endian)
 *    If 2^255-MQ is not prime, GF_MODPRIME must be zero
 *
 * GF_MODPRIME, when non-zero, removes the final check of inversion success;
 * this check is harmless but unnecessary when the modulus is prime.
 *
 * Exception: gf_inv_FLT(), if defined, uses an hardcoded addition chain
 * for 2^255-19, and will not work properly for other moduli (prime or not).
 */

#define MQ   19
static const gf GF_INVT508 = {
	0x29D6DEAB9CB8606C,
	0x788DD408827B63FD,
	0x3CFC744C965683E6,
	0x24E016B1490AA31A
};
#define GF_MODPRIME   1

/*
 * Constants below need not be modified if MQ is changed.
 */
static const gf GF_ZERO = { 0, 0, 0, 0 };
static const gf GF_ONE = { 1, 0, 0, 0 };
static const gf GF_P = {
	-MQ, (uint64_t)-1, (uint64_t)-1, (uint64_t)-1 >> 1
};

/*
 * Helpful macro to convert expressions to strings, for inline assembly.
 */
#define ST(x)    ST_(x)
#define ST_(x)   #x

/*
 * If USE_ALT_REDUCTION is 1, then an alternate reduction step is used
 * in gf_mul_inline() and gf_sqr_inline(). The alternate reduction step
 * involves an extra multiplication, but reduces the length of a
 * dependency path in case of repeated squarings; it is used
 * systematically in gf_sqr_x_inline(), where it appears to save 2
 * cycles per squaring. For non-repeated squarings and for multiplications,
 * it may have adverse effects depending on where it is used (specifically,
 * how the output values are used), hence we keep the possibility of
 * disabling it.
 */
#ifndef USE_ALT_REDUCTION
#define USE_ALT_REDUCTION   1
#endif

/*
 * A field element is represented as four limbs, in base 2^64. Operands
 * and result may be up to 2^256-1.
 * A _normalized_ value is in 0..p-1. gf_normalize() ensures normalization;
 * it is called before encoding, and also when starting inversion.
 */

/* see gf25519.h */
void
gf_add(gf *d, const gf *a, const gf *b)
{
	unsigned long long d0, d1, d2, d3;
	unsigned char cc;

	/*
	 * Compute sum over 256 bits + carry.
	 */
	cc = _addcarry_u64(0, a->v0, b->v0, &d0);
	cc = _addcarry_u64(cc, a->v1, b->v1, &d1);
	cc = _addcarry_u64(cc, a->v2, b->v2, &d2);
	cc = _addcarry_u64(cc, a->v3, b->v3, &d3);

	/*
	 * If there is a carry, subtract 2*p, i.e. add 2*MQ and
	 * (implicitly) subtract 2^256.
	 */
	cc = _addcarry_u64(0, d0, -(unsigned long long)cc & (2 * MQ), &d0);
	cc = _addcarry_u64(cc, d1, 0, (unsigned long long *)&d->v1);
	cc = _addcarry_u64(cc, d2, 0, (unsigned long long *)&d->v2);
	cc = _addcarry_u64(cc, d3, 0, (unsigned long long *)&d->v3);

	/*
	 * Usually, there won't be an extra carry here, and the
	 * implicit subtraction of 2^256 is enough. However, if the
	 * the original sum was at least 2^257 - 2*MQ, then this
	 * will show up as an extra carry here. In that case, the
	 * low limb (d0) is necessarily less than 2*MQ, and thus adding
	 * back 2*MQ to it will not trigger any carry.
	 */
	(void)_addcarry_u64(0, d0, -(unsigned long long)cc & (2 * MQ),
		(unsigned long long *)&d->v0);
}

/* see gf25519.h */
void
gf_sub(gf *d, const gf *a, const gf *b)
{
	unsigned long long d0, d1, d2, d3, e;
	unsigned char cc;

	/*
	 * Compute subtraction over 256 bits.
	 */
	cc = _subborrow_u64(0, a->v0, b->v0, &d0);
	cc = _subborrow_u64(cc, a->v1, b->v1, &d1);
	cc = _subborrow_u64(cc, a->v2, b->v2, &d2);
	cc = _subborrow_u64(cc, a->v3, b->v3, &d3);

	/*
	 * If the result is negative, add back 2*p = 2^256 - 2*MQ (which
	 * we do by subtracting 2*MQ).
	 */
	e = -(unsigned long long)cc;
	cc = _subborrow_u64(0, d0, e & (2 * MQ), &d0);
	cc = _subborrow_u64(cc, d1, 0, (unsigned long long *)&d->v1);
	cc = _subborrow_u64(cc, d2, 0, (unsigned long long *)&d->v2);
	cc = _subborrow_u64(cc, d3, 0, (unsigned long long *)&d->v3);

	/*
	 * If there is still a carry, then we must add 2*p again. In that
	 * case, subtracting 2*MQ from the low limb won't trigger a carry.
	 */
	d->v0 = d0 - (-(unsigned long long)cc & (2 * MQ));
}

/* see gf25519.h */
void
gf_neg(gf *d, const gf *a)
{
	unsigned long long d0, d1, d2, d3, e;
	unsigned char cc;

	/*
	 * Compute 2*p - a over 256 bits.
	 */
	cc = _subborrow_u64(0, (unsigned long long)-(2 * MQ), a->v0, &d0);
	cc = _subborrow_u64(cc, (unsigned long long)-1, a->v1, &d1);
	cc = _subborrow_u64(cc, (unsigned long long)-1, a->v2, &d2);
	cc = _subborrow_u64(cc, (unsigned long long)-1, a->v3, &d3);

	/*
	 * If the result is negative, add back p = 2^255 - MQ.
	 */
	e = -(unsigned long long)cc;
	cc = _addcarry_u64(0, d0, e & -MQ, (unsigned long long *)&d->v0);
	cc = _addcarry_u64(cc, d1, e, (unsigned long long *)&d->v1);
	cc = _addcarry_u64(cc, d2, e, (unsigned long long *)&d->v2);
	(void)_addcarry_u64(cc, d3, e >> 1, (unsigned long long *)&d->v3);
}

/* see gf25519.h */
void
gf_condneg(gf *d, const gf *a, uint64_t ctl)
{
	gf t;

	gf_neg(&t, a);
	d->v0 = a->v0 ^ (-ctl & (a->v0 ^ t.v0));
	d->v1 = a->v1 ^ (-ctl & (a->v1 ^ t.v1));
	d->v2 = a->v2 ^ (-ctl & (a->v2 ^ t.v2));
	d->v3 = a->v3 ^ (-ctl & (a->v3 ^ t.v3));
}

#if 0
/* gf_mul() with no assembly, only 'unsigned __int128'. */

#define M64(lo, hi, x, y)   do { \
		unsigned __int128 m64_tmp; \
		m64_tmp = (unsigned __int128)(x) * (unsigned __int128)(y); \
		(lo) = (unsigned long long)m64_tmp; \
		(hi) = (unsigned long long)(m64_tmp >> 64); \
	} while (0)

#define M64_ADD(lo, hi, x, y, t1)   do { \
		unsigned __int128 m64_tmp; \
		m64_tmp = (unsigned __int128)(x) * (unsigned __int128)(y) \
			+ (unsigned __int128)(t1); \
		(lo) = (unsigned long long)m64_tmp; \
		(hi) = (unsigned long long)(m64_tmp >> 64); \
	} while (0)

#define M64_ADD2(lo, hi, x, y, t1, t2)   do { \
		unsigned __int128 m64_tmp; \
		m64_tmp = (unsigned __int128)(x) * (unsigned __int128)(y) \
			+ (unsigned __int128)(t1) + (unsigned __int128)(t2); \
		(lo) = (unsigned long long)m64_tmp; \
		(hi) = (unsigned long long)(m64_tmp >> 64); \
	} while (0)

void
gf_mul(gf *d, const gf *a, const gf *b)
{
	unsigned long long t0, t1, t2, t3, t4, t5, t6, t7;
	unsigned long long m, h;
	unsigned char cc;

	/*
	 * Compute 512-bit product in t0..t7.
	 */
	m = a->v0;
	M64(t0, h, m, b->v0);
	M64_ADD(t1, h, m, b->v1, h);
	M64_ADD(t2, h, m, b->v2, h);
	M64_ADD(t3, t4, m, b->v3, h);

	m = a->v1;
	M64_ADD(t1, h, m, b->v0, t1);
	M64_ADD2(t2, h, m, b->v1, t2, h);
	M64_ADD2(t3, h, m, b->v2, t3, h);
	M64_ADD2(t4, t5, m, b->v3, t4, h);

	m = a->v2;
	M64_ADD(t2, h, m, b->v0, t2);
	M64_ADD2(t3, h, m, b->v1, t3, h);
	M64_ADD2(t4, h, m, b->v2, t4, h);
	M64_ADD2(t5, t6, m, b->v3, t5, h);

	m = a->v3;
	M64_ADD(t3, h, m, b->v0, t3);
	M64_ADD2(t4, h, m, b->v1, t4, h);
	M64_ADD2(t5, h, m, b->v2, t5, h);
	M64_ADD2(t6, t7, m, b->v3, t6, h);

	/*
	 * Reduction, first pass.
	 */
	m = 2 * MQ;
	M64_ADD(t0, h, m, t4, t0);
	M64_ADD2(t1, h, m, t5, t1, h);
	M64_ADD2(t2, h, m, t6, t2, h);
	M64_ADD2(t3, t4, m, t7, t3, h);

	/*
	 * Reduction, second pass. Since there's only a single word (t4),
	 * most of it is additions. Note that t4 < 2*MQ, and 4*MQ^2 < 2^64;
	 * therefore, we can use a simple multiplication on 64 bits.
	 */
	cc = _addcarry_u64(0, t0, m * t4, &t0);
	cc = _addcarry_u64(cc, t1, 0, (unsigned long long *)&d->v1);
	cc = _addcarry_u64(cc, t2, 0, (unsigned long long *)&d->v2);
	cc = _addcarry_u64(cc, t3, 0, (unsigned long long *)&d->v3);

	/*
	 * If there is a remaining carry here, then we have to propagate
	 * it back. But in that case, t0 < 4*MQ^2 and adding 2*MQ cannot
	 * overflow (4*MQ^2 + 2*MQ < 2^64).
	 */
	d->v0 = t0 + (-(uint64_t)cc & (2 * MQ));
}
#endif

__attribute__((used))
static const uint64_t global_m63 = 0x7FFFFFFFFFFFFFFF;

/*
 * Inline version of multiplication, using assembly.
 */
__attribute__((always_inline))
static inline void
gf_mul_inline(gf *d, const gf *a, const gf *b)
{
	__asm__ __volatile__ (
		/*
		 * We compute the 512-bit result into r8..r15. Carry
		 * word is in rax. Multiplier word is in rdx (since that's
		 * what mulx requires). Note that mulx does not affect
		 * flags.
		 */

		/* Clear r15. This also clears CF and OF */
		"xorq	%%r15, %%r15\n\t"

		/* a0*b -> r8..r12 */
		"movq	(%%rsi), %%rdx\n\t"
		"mulx	(%%rcx), %%r8, %%r9\n\t"
		"mulx	8(%%rcx), %%rax, %%r10\n\t"
		"adox	%%rax, %%r9\n\t"
		"mulx	16(%%rcx), %%rax, %%r11\n\t"
		"adox	%%rax, %%r10\n\t"
		"mulx	24(%%rcx), %%rax, %%r12\n\t"
		"adox	%%rax, %%r11\n\t"
		"adox	%%r15, %%r12\n\t"

		/* This last adox cannot overflow, and thus OF=0 */

		/* a1*b -> added to r9..r13 */
		"movq	8(%%rsi), %%rdx\n\t"
		"mulx	(%%rcx), %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r9\n\t"
		"adox	%%rbx, %%r10\n\t"
		"mulx	8(%%rcx), %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r10\n\t"
		"adox	%%rbx, %%r11\n\t"
		"mulx	16(%%rcx), %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r11\n\t"
		"adox	%%rbx, %%r12\n\t"
		"mulx	24(%%rcx), %%rax, %%r13\n\t"
		"adcx	%%rax, %%r12\n\t"
		"adox	%%r15, %%r13\n\t"
		"adcx	%%r15, %%r13\n\t"

		/* Again, the last adcx and adox cannot overflow,
		   therefore CF=0 and OF=0 at this point */

		/* a2*b -> added to r10..r14 */
		"movq	16(%%rsi), %%rdx\n\t"
		"mulx	(%%rcx), %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r10\n\t"
		"adox	%%rbx, %%r11\n\t"
		"mulx	8(%%rcx), %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r11\n\t"
		"adox	%%rbx, %%r12\n\t"
		"mulx	16(%%rcx), %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r12\n\t"
		"adox	%%rbx, %%r13\n\t"
		"mulx	24(%%rcx), %%rax, %%r14\n\t"
		"adcx	%%rax, %%r13\n\t"
		"adox	%%r15, %%r14\n\t"
		"adcx	%%r15, %%r14\n\t"

		/* a3*b -> added to r11..r15 */
		"movq	24(%%rsi), %%rdx\n\t"
		"xorq	%%rsi, %%rsi\n\t"
		"mulx	(%%rcx), %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r11\n\t"
		"adox	%%rbx, %%r12\n\t"
		"mulx	8(%%rcx), %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r12\n\t"
		"adox	%%rbx, %%r13\n\t"
		"mulx	16(%%rcx), %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r13\n\t"
		"adox	%%rbx, %%r14\n\t"
		"mulx	24(%%rcx), %%rax, %%r15\n\t"
		"adcx	%%rax, %%r14\n\t"
		"adox	%%rsi, %%r15\n\t"
		"adcx	%%rsi, %%r15\n\t"

		/* Multiply high words by 2*MQ and fold them on low words. */
		"movl	$(" ST(2 * MQ) "), %%edx\n\t"
		"mulx	%%r12, %%rax, %%r12\n\t"
		"adcx	%%rax, %%r8\n\t"
		"adox	%%r12, %%r9\n\t"
		"mulx	%%r13, %%rax, %%r13\n\t"
		"adcx	%%rax, %%r9\n\t"
		"adox	%%r13, %%r10\n\t"
		"mulx	%%r14, %%rax, %%r14\n\t"
		"adcx	%%rax, %%r10\n\t"
		"adox	%%r14, %%r11\n\t"
		"mulx	%%r15, %%rax, %%r15\n\t"
		"adcx	%%rax, %%r11\n\t"
		"adox	%%rsi, %%r15\n\t"
		"adcx	%%rsi, %%r15\n\t"

#if USE_ALT_REDUCTION
		/* Augment extra word with top bit, and multiply by MQ
		   to fold. This is extra work right now, but removes
		   an extra dependency later on. */
		"shld	$1, %%r11, %%r15\n\t"
		"andq	global_m63(%%rip), %%r11\n\t"
		"imulq	$(" ST(MQ) "), %%r15, %%r15\n\t"
		"addq	%%r15, %%r8\n\t"
		"adcq	%%rsi, %%r9\n\t"
		"adcq	%%rsi, %%r10\n\t"
		"adcq	%%rsi, %%r11\n\t"

		/* There cannot be a carry here, because top limb was
		   only 63 bits. */
#else
		/* Extra word is in r15, multiply by 2*MQ and fold. */
		"mulx	%%r15, %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r8\n\t"
		"adcx	%%rsi, %%r9\n\t"
		"adcx	%%rsi, %%r10\n\t"
		"adcx	%%rsi, %%r11\n\t"

		/* If there is still a carry, fold it again. This time,
		   there cannot be a carry beyond the first limb. */
		"cmovc	%%rdx, %%rsi\n\t"
		"addq	%%rsi, %%r8\n\t"
#endif

		/* Write result into output structure. */
		"movq	%%r8, (%0)\n\t"
		"movq	%%r9, 8(%0)\n\t"
		"movq	%%r10, 16(%0)\n\t"
		"movq	%%r11, 24(%0)\n\t"

		: "=D" (d), "=S" (a), "=c" (b)
		: "0" (d), "1" (a), "2" (b)
		: "cc", "memory", "rax", "rbx", "rdx", "r8", "r9",
		  "r10", "r11", "r12", "r13", "r14", "r15"
	);
}

/* see gf25519.h */
__attribute__((noinline))
void
gf_mul(gf *d, const gf *a, const gf *b)
{
	gf_mul_inline(d, a, b);
}

/*
 * Inline version of squarings, using assembly.
 */
__attribute__((always_inline))
static inline void
gf_sqr_inline(gf *d, const gf *a)
{
	__asm__ __volatile__ (
		/*
		 * We compute the 512-bit result into r8..r15. Carry
		 * word is in rax. Multiplier word is in rdx (since that's
		 * what mulx requires). Note that mulx does not affect
		 * flags.
		 */

		/* Load a0 into rdx */
		"movq	(%1), %%rdx\n\t"

		/* Clear r15, CF and OF. */
		"xorl	%%r15d, %%r15d\n\t"

		/* a0*a1 -> r9:r10
		   a0*a2 -> r10:r11
		   a0*a3 -> r11:r12 */
		"mulx	8(%1), %%r9, %%r10\n\t"
		"mulx	16(%1), %%r13, %%r11\n\t"
		"mulx	24(%1), %%r14, %%r12\n\t"
		"adcx	%%r13, %%r10\n\t"    // CF for r11
		"adox	%%r14, %%r11\n\t"    // OF for r12

		/* a1*a2 -> r11:r12
		   a1*a3 -> r12:r13 */
		"movq	8(%1), %%rdx\n\t"
		"mulx	16(%1), %%rax, %%rbx\n\t"
		"mulx	24(%1), %%rcx, %%r13\n\t"
		"adcx	%%rax, %%r11\n\t"    // CF for r12
		"adox	%%rbx, %%r12\n\t"    // OF for r13
		"adcx	%%rcx, %%r12\n\t"    // CF for r13

		/* a2*a3 -> r13:r14 */
		"movq	16(%1), %%rdx\n\t"
		"mulx	24(%1), %%rax, %%r14\n\t"
		"adox	%%rax, %%r13\n\t"    // OF for r14
		"adcx	%%r15, %%r13\n\t"    // CF for r14
		"adox	%%r15, %%r14\n\t"    // OF=0
		"adcx	%%r15, %%r14\n\t"    // CF=0

		/* We have the cross products in place, but we must
		   double them. This is a shift, and we use shl/shld
		   since it avoids carry propagation delays. */
		"shld	$1, %%r14, %%r15\n\t"
		"shld	$1, %%r13, %%r14\n\t"
		"shld	$1, %%r12, %%r13\n\t"
		"shld	$1, %%r11, %%r12\n\t"
		"shld	$1, %%r10, %%r11\n\t"
		"shld	$1, %%r9, %%r10\n\t"
		"addq	%%r9, %%r9\n\t"

		/* Clear CF and OF. r8 was not used yet. */
		"xorl	%%r8d, %%r8d\n\t"

		/* Now add a0^2, a1^2, a2^2 and a3^3
		   We still have a2 in rdx */
		"mulx	%%rdx, %%rax, %%rbx\n\t"
		"movq	(%1), %%rdx\n\t"
		"mulx	%%rdx, %%r8, %%rcx\n\t"
		"adcx	%%rcx, %%r9\n\t"
		"movq	8(%1), %%rdx\n\t"
		"mulx	%%rdx, %%rdx, %%rcx\n\t"
		"adcx	%%rdx, %%r10\n\t"
		"adcx	%%rcx, %%r11\n\t"
		"adcx	%%rax, %%r12\n\t"
		"adcx	%%rbx, %%r13\n\t"
		"movq	24(%1), %%rdx\n\t"
		"mulx	%%rdx, %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r14\n\t"
		"adcx	%%rbx, %%r15\n\t"

		/* At that point, CF=0 and OF=0.
		   We have the 512-bit square in r8..r15. */

		/* Clear rsi. */
		"xorl	%%esi, %%esi\n\t"

		/* Multiply high words by 2*MQ and fold them on low words. */
		"movl	$(" ST(2 * MQ) "), %%edx\n\t"
		"mulx	%%r12, %%rax, %%r12\n\t"
		"adcx	%%rax, %%r8\n\t"
		"adox	%%r12, %%r9\n\t"
		"mulx	%%r13, %%rax, %%r13\n\t"
		"adcx	%%rax, %%r9\n\t"
		"adox	%%r13, %%r10\n\t"
		"mulx	%%r14, %%rax, %%r14\n\t"
		"adcx	%%rax, %%r10\n\t"
		"adox	%%r14, %%r11\n\t"
		"mulx	%%r15, %%rax, %%r15\n\t"
		"adcx	%%rax, %%r11\n\t"
		"adox	%%rsi, %%r15\n\t"
		"adcx	%%rsi, %%r15\n\t"

#if USE_ALT_REDUCTION
		/* Augment extra word with top bit, and multiply by MQ
		   to fold. This is extra work right now, but removes
		   an extra dependency later on. */
		"shld	$1, %%r11, %%r15\n\t"
		"andq	global_m63(%%rip), %%r11\n\t"
		"imulq	$(" ST(MQ) "), %%r15, %%r15\n\t"
		"addq	%%r15, %%r8\n\t"
		"adcq	%%rsi, %%r9\n\t"
		"adcq	%%rsi, %%r10\n\t"
		"adcq	%%rsi, %%r11\n\t"

		/* There cannot be a carry here, because top limb was
		   only 63 bits. */
#else
		/* Extra word is in r15, multiply by 2*MQ and fold. */
		"mulx	%%r15, %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r8\n\t"
		"adcx	%%rsi, %%r9\n\t"
		"adcx	%%rsi, %%r10\n\t"
		"adcx	%%rsi, %%r11\n\t"

		/* If there is still a carry, fold it again. This time,
		   there cannot be a carry beyond the first limb. */
		"cmovc	%%rdx, %%rsi\n\t"
		"addq	%%rsi, %%r8\n\t"
#endif

		/* Write result into output structure. */
		"movq	%%r8, (%0)\n\t"
		"movq	%%r9, 8(%0)\n\t"
		"movq	%%r10, 16(%0)\n\t"
		"movq	%%r11, 24(%0)\n\t"

		: "=D" (d), "=S" (a)
		: "0" (d), "1" (a)
		: "cc", "memory", "rax", "rbx", "rcx", "rdx", "r8", "r9",
		  "r10", "r11", "r12", "r13", "r14", "r15"
	);
}

/* see gf25519.h */
__attribute__((noinline))
void
gf_sqr(gf *d, const gf *a)
{
	gf_sqr_inline(d, a);
}

/*
 * Inline version of the multiple squarings function, using assembly.
 */
__attribute__((always_inline))
static void
gf_sqr_x_inline(gf *d, const gf *a, long num)
{
	__asm__ __volatile__ (
		/*
		 * Load a0..a3 into rax:rbx:rcx:rbp
		 * Then reuse esi as loop counter.
		 */
		"movq	(%%rsi), %%rax\n\t"
		"movq	8(%%rsi), %%rbx\n\t"
		"movq	16(%%rsi), %%rcx\n\t"
		"movq	24(%%rsi), %%rbp\n\t"
		"movl	%%edx, %%esi\n\t"

		/* Loop entry. */
		"0:\n\t"

		/* Load a0 into rdx */
		"movq	%%rax, %%rdx\n\t"

		/* Clear r15, CF and OF. */
		"xorl	%%r15d, %%r15d\n\t"

		/* a0*a1 -> r9:r10
		   a0*a2 -> r10:r11
		   a0*a3 -> r11:r12 */
		"mulx	%%rbx, %%r9, %%r10\n\t"
		"mulx	%%rcx, %%r13, %%r11\n\t"
		"mulx	%%rbp, %%r14, %%r12\n\t"
		"adcx	%%r13, %%r10\n\t"    // CF for r11
		"adox	%%r14, %%r11\n\t"    // OF for r12

		/* a1*a2 -> r11:r12
		   a1*a3 -> r12:r13 */
		"movq	%%rbx, %%rdx\n\t"
		"mulx	%%rcx, %%r8, %%r14\n\t"
		"adcx	%%r8, %%r11\n\t"     // CF for r12
		"mulx	%%rbp, %%r8, %%r13\n\t"
		"adox	%%r14, %%r12\n\t"    // OF for r13
		"adcx	%%r8, %%r12\n\t"     // CF for r13

		/* a2*a3 -> r13:r14 */
		"movq	%%rcx, %%rdx\n\t"
		"mulx	%%rbp, %%r8, %%r14\n\t"
		"adox	%%r8, %%r13\n\t"     // OF for r14
		"adcx	%%r15, %%r13\n\t"    // CF for r14
		"adox	%%r15, %%r14\n\t"    // OF=0
		"adcx	%%r15, %%r14\n\t"    // CF=0

		/* We have the cross products in place, but we must
		   double them. */
		"shld	$1, %%r14, %%r15\n\t"
		"shld	$1, %%r13, %%r14\n\t"
		"shld	$1, %%r12, %%r13\n\t"
		"shld	$1, %%r11, %%r12\n\t"
		"shld	$1, %%r10, %%r11\n\t"
		"shld	$1, %%r9, %%r10\n\t"
		"addq	%%r9, %%r9\n\t"

		/* Clear CF and OF. r8 is still scratch at this point. */
		"xorl	%%r8d, %%r8d\n\t"

		/* Now add a0^2, a1^2, a2^2 and a3^2 */
		"movq	%%rax, %%rdx\n\t"
		"mulx	%%rax, %%r8, %%rax\n\t"
		"adcx	%%rax, %%r9\n\t"
		"movq	%%rbx, %%rdx\n\t"
		"mulx	%%rbx, %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r10\n\t"
		"adcx	%%rbx, %%r11\n\t"
		"movq	%%rcx, %%rdx\n\t"
		"mulx	%%rcx, %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r12\n\t"
		"adcx	%%rbx, %%r13\n\t"
		"movq	%%rbp, %%rdx\n\t"
		"mulx	%%rbp, %%rax, %%rbx\n\t"
		"adcx	%%rax, %%r14\n\t"
		"adcx	%%rbx, %%r15\n\t"

		/* At that point, CF=0 and OF=0.
		   We have the 512-bit square in r8..r15. */

		/* Multiply high words by 2*MQ and fold them on low words.
		   At the same time, we move the resulting low words into
		   rax:rbx:rcx:rbp */
		"movl	$(" ST(2 * MQ) "), %%edx\n\t"
		"mulx	%%r12, %%rax, %%r12\n\t"
		"adcx	%%r8, %%rax\n\t"
		"adox	%%r12, %%r9\n\t"
		"mulx	%%r13, %%rbx, %%r13\n\t"
		"adcx	%%r9, %%rbx\n\t"
		"adox	%%r13, %%r10\n\t"
		"mulx	%%r14, %%rcx, %%r14\n\t"
		"adcx	%%r10, %%rcx\n\t"
		"movl	$0, %%r10d\n\t"
		"adox	%%r14, %%r11\n\t"
		"mulx	%%r15, %%rbp, %%r15\n\t"
		"adcx	%%r11, %%rbp\n\t"
		"adox	%%r10, %%r15\n\t"
		"adcx	%%r10, %%r15\n\t"

		/* Augment extra word with top bit, and multiply by MQ
		   to fold. This is extra work right now, but removes
		   an extra dependency later on. */
		"shld	$1, %%rbp, %%r15\n\t"
		"andq	global_m63(%%rip), %%rbp\n\t"
		"imulq	$(" ST(MQ) "), %%r15, %%r15\n\t"
		"addq	%%r15, %%rax\n\t"
		"adcq	%%r10, %%rbx\n\t"
		"adcq	%%r10, %%rcx\n\t"
		"adcq	%%r10, %%rbp\n\t"

		/* There cannot be a carry here, because top limb was
		   only 63 bits. */

		/*
		 * Loop until all squares have been performed.
		 */
		"decl	%%esi\n\t"
		"jnz	0b\n\t"

		/*
		 * Write result.
		 */
		"movq	%%rax, (%%rdi)\n\t"
		"movq	%%rbx, 8(%%rdi)\n\t"
		"movq	%%rcx, 16(%%rdi)\n\t"
		"movq	%%rbp, 24(%%rdi)\n\t"

		: "=D" (d), "=S" (a), "=d" (num)
		: "0" (d), "1" (a), "2" (num)
		: "cc", "memory", "rax", "rbx", "rcx", "rbp", "r8", "r9",
		  "r10", "r11", "r12", "r13", "r14", "r15"
	);
}

/* see gf25519.h */
__attribute__((noinline))
void
gf_sqr_x(gf *d, const gf *a, long num)
{
	gf_sqr_x_inline(d, a, num);
}

/*
 * Normalize a value to 0..p-1.
 */
static void
gf_normalize(gf *d, const gf *a)
{
	unsigned long long d0, d1, d2, d3, e;
	unsigned char cc;

	/*
	 * If top bit is set, propagate it.
	 */
	e = -(unsigned long long)(a->v3 >> 63);
	cc = _addcarry_u64(0, a->v0, e & MQ, &d0);
	cc = _addcarry_u64(cc, a->v1, 0, &d1);
	cc = _addcarry_u64(cc, a->v2, 0, &d2);
	(void)_addcarry_u64(cc, a->v3, e << 63, &d3);

	/*
	 * Value is now at most 2^255+MQ-1. Subtract p, and add it back
	 * if the result is negative.
	 */
	cc = _subborrow_u64(0, d0, (unsigned long long)-MQ, &d0);
	cc = _subborrow_u64(cc, d1, (unsigned long long)-1, &d1);
	cc = _subborrow_u64(cc, d2, (unsigned long long)-1, &d2);
	cc = _subborrow_u64(cc, d3, (unsigned long long)-1 >> 1, &d3);

	e = -(unsigned long long)cc;
	cc = _addcarry_u64(0, d0, e & -MQ, (unsigned long long *)&d->v0);
	cc = _addcarry_u64(cc, d1, e, (unsigned long long *)&d->v1);
	cc = _addcarry_u64(cc, d2, e, (unsigned long long *)&d->v2);
	(void)_addcarry_u64(cc, d3, e >> 1, (unsigned long long *)&d->v3);
}

/* see gf25519.h */
uint64_t
gf_iszero(const gf *a)
{
	unsigned long long t0, t1, t2;

	/*
	 * Since values are over 256 bits, there are three possible
	 * representations for 0: 0, p and 2*p.
	 */
	t0 = a->v0 | a->v1 | a->v2 | a->v3;
	t1 = (a->v0 + MQ) | ~a->v1 | ~a->v2 | (a->v3 ^ 0x7FFFFFFFFFFFFFFF);
	t2 = (a->v0 + (2 * MQ)) | ~a->v1 | ~a->v2 | ~a->v3;

	/*
	 * Value is zero if and only if one of t0, t1 or t2 is zero.
	 */
	return 1 - (((t0 | -t0) & (t1 | -t1) & (t2 | -t2)) >> 63);
}

/* see gf25519.h */
uint64_t
gf_eq(const gf *a, const gf *b)
{
	gf t;

	gf_sub(&t, a, b);
	return gf_iszero(&t);
}

#if defined GF_INV_FLT && GF_INV_FLT

/*
 * If GF_INV_FLT is at least 2, then override calls to gf_mul(),
 * gf_sqr() and gf_sqr_x() to use the inline versions. This makes the
 * resulting binary quite larger, but also slightly faster (not fast
 * enough to compete with gf_inv(), though).
 */
#if GF_INV_FLT >= 2
#define gf_sqr     gf_sqr_inline
#define gf_sqr_x   gf_sqr_x_inline
#define gf_mul     gf_mul_inline
#endif

/* see gf25519.h */
uint64_t
gf_inv_FLT(gf *d, const gf *a)
{
	/*
	 * This code works only for p = 2^255-19. Inversion is done
	 * by raising to the power p-2 = 2^255-21; since p is prime,
	 * this yields the right result (and it yields 0 when input
	 * is 0, which is the expected behaviour in that case).
	 *
	 * For the exponent, an addition chain is used that uses
	 * 254 squarings and 11 extra multiplications. This is the
	 * addition chain from the original Curve25519 reference code.
	 */
	gf x, y, z, t;

	/* x <- a^2 */
	gf_sqr(&x, a);

	/* y <- a^9 */
	gf_sqr_x(&y, &x, 2);
	gf_mul(&y, &y, a);

	/* x <- a^11 */
	gf_mul(&x, &y, &x);

	/* y <- a^31 */
	gf_sqr(&z, &x);
	gf_mul(&y, &y, &z);

	/* y <- a^1023 = a^(2^10-1) */
	gf_sqr_x(&z, &y, 5);
	gf_mul(&y, &z, &y);

	/* z <- a^(2^20-1) */
	gf_sqr_x(&z, &y, 10);
	gf_mul(&z, &z, &y);

	/* z <- a^(2^40-1) */
	gf_sqr_x(&t, &z, 20);
	gf_mul(&z, &z, &t);

	/* z <- a^(2^50-1) */
	gf_sqr_x(&z, &z, 10);
	gf_mul(&z, &z, &y);

	/* y <- a^(2^100-1) */
	gf_sqr_x(&y, &z, 50);
	gf_mul(&y, &y, &z);

	/* y <- a^(2^200-1) */
	gf_sqr_x(&t, &y, 100);
	gf_mul(&y, &y, &t);

	/* y <- a^(2^250-1) */
	gf_sqr_x(&y, &y, 50);
	gf_mul(&y, &y, &z);

	/* d <- a^(2^255-19) */
	gf_sqr_x(&y, &y, 5);
	gf_mul(d, &y, &x);

	/* If value was invertible, we get a non-zero here. If value
	   was not invertible, it was zero, and we get a zero result. */
	return gf_iszero(d) ^ 1;
}

/*
 * Remove overrides for force-inlining (if defined previously).
 */
#undef gf_sqr
#undef gf_sqr_x
#undef gf_mul

#endif

/*
 * Compute (a*f+b*g)/2^31. Parameters f and g are provided with an
 * unsigned type, but they are signed integers in the -2^31..+2^31
 * range. Values a, b and d are not field elements, but signed 256-bit
 * integers (i.e. top bit is the sign bit) which are nonnegative (value
 * is between 0 and 2^255-1). The division by 2^31 is assumed to be
 * exact (low j bits of a*f+b*g are dropped). The result is assumed to
 * fit in 256 bits, including the sign bit (truncation is applied on
 * higher bits).
 *
 * If the result turns out to be negative, then it is negated. Returned
 * value is 1 if the result was negated, 0 otherwise.
 */
static inline uint64_t
s256_lin_div31_abs(gf *d, const gf *a, const gf *b,
	unsigned long long f, unsigned long long g)
{
	gf ta, tb;
	unsigned long long sf, sg, d0, d1, d2, d3, t;
	unsigned __int128 z;
	unsigned char cc;

	/*
	 * If f < 0, replace f with -f but keep the sign in sf.
	 * Similarly for g.
	 */
	sf = f >> 63;
	f = (f ^ -sf) + sf;
	sg = g >> 63;
	g = (g ^ -sg) + sg;

	/*
	 * Apply signs sf and sg to a and b, respectively.
	 */
	cc = _addcarry_u64(0, a->v0 ^ -sf, sf, (unsigned long long *)&ta.v0);
	cc = _addcarry_u64(cc, a->v1 ^ -sf, 0, (unsigned long long *)&ta.v1);
	cc = _addcarry_u64(cc, a->v2 ^ -sf, 0, (unsigned long long *)&ta.v2);
	cc = _addcarry_u64(cc, a->v3 ^ -sf, 0, (unsigned long long *)&ta.v3);

	cc = _addcarry_u64(0, b->v0 ^ -sg, sg, (unsigned long long *)&tb.v0);
	cc = _addcarry_u64(cc, b->v1 ^ -sg, 0, (unsigned long long *)&tb.v1);
	cc = _addcarry_u64(cc, b->v2 ^ -sg, 0, (unsigned long long *)&tb.v2);
	cc = _addcarry_u64(cc, b->v3 ^ -sg, 0, (unsigned long long *)&tb.v3);

	/*
	 * Now that f and g are nonnegative, compute a*f+b*g into
	 * d0:d1:d2:d3:t. Since f and g are at most 2^31, we can
	 * add two 128-bit products with no overflow (they are actually
	 * 95 bits each at most).
	 */
	z = (unsigned __int128)ta.v0 * (unsigned __int128)f
		+ (unsigned __int128)tb.v0 * (unsigned __int128)g;
	d0 = (unsigned long long)z;
	t = (unsigned long long)(z >> 64);
	z = (unsigned __int128)ta.v1 * (unsigned __int128)f
		+ (unsigned __int128)tb.v1 * (unsigned __int128)g
		+ (unsigned __int128)t;
	d1 = (unsigned long long)z;
	t = (unsigned long long)(z >> 64);
	z = (unsigned __int128)ta.v2 * (unsigned __int128)f
		+ (unsigned __int128)tb.v2 * (unsigned __int128)g
		+ (unsigned __int128)t;
	d2 = (unsigned long long)z;
	t = (unsigned long long)(z >> 64);
	z = (unsigned __int128)ta.v3 * (unsigned __int128)f
		+ (unsigned __int128)tb.v3 * (unsigned __int128)g
		+ (unsigned __int128)t;
	d3 = (unsigned long long)z;
	t = (unsigned long long)(z >> 64);

	/*
	 * Don't forget the signs: if a < 0, then the result is
	 * overestimated by 2^256*f; similarly, if b < 0, then the
	 * result is overestimated by 2^256*g. We thus must subtract
	 * 2^256*(sa*f+sb*g), where sa and sb are the signs of a and b,
	 * respectively.
	 */
	t -= -(unsigned long long)(ta.v3 >> 63) & f;
	t -= -(unsigned long long)(tb.v3 >> 63) & g;

	/*
	 * Apply the shift.
	 */
	d0 = (d0 >> 31) | (d1 << 33);
	d1 = (d1 >> 31) | (d2 << 33);
	d2 = (d2 >> 31) | (d3 << 33);
	d3 = (d3 >> 31) | (t << 33);

	/*
	 * Perform conditional negation, if the result is negative.
	 */
	t >>= 63;
	cc = _addcarry_u64(0, d0 ^ -t, t, (unsigned long long *)&d->v0);
	cc = _addcarry_u64(cc, d1 ^ -t, 0, (unsigned long long *)&d->v1);
	cc = _addcarry_u64(cc, d2 ^ -t, 0, (unsigned long long *)&d->v2);
	(void)_addcarry_u64(cc, d3 ^ -t, 0, (unsigned long long *)&d->v3);

	return t;
}

/*
 * Compute u*f+v*g (modulo p). Parameters f and g are provided with
 * an unsigned type, but they are signed integers in the -2^62..+2^62 range.
 */
static inline void
gf_lin(gf *d, const gf *u, const gf *v,
	unsigned long long f, unsigned long long g)
{
	gf tu, tv;
	unsigned long long sf, sg, d0, d1, d2, d3, t;
	unsigned __int128 z;
	unsigned char cc;

	/*
	 * If f < 0, replace f with -f but keep the sign in sf.
	 * Similarly for g.
	 */
	sf = f >> 63;
	f = (f ^ -sf) + sf;
	sg = g >> 63;
	g = (g ^ -sg) + sg;

	/*
	 * Apply signs sf and sg to u and v.
	 */
	gf_condneg(&tu, u, sf);
	gf_condneg(&tv, v, sg);

	/*
	 * Since f <= 2^62 and g <= 2^62, we can add 128-bit products: they
	 * fit on 126 bits each.
	 */
	z = (unsigned __int128)tu.v0 * (unsigned __int128)f
		+ (unsigned __int128)tv.v0 * (unsigned __int128)g;
	d0 = (unsigned long long)z;
	t = (unsigned long long)(z >> 64);
	z = (unsigned __int128)tu.v1 * (unsigned __int128)f
		+ (unsigned __int128)tv.v1 * (unsigned __int128)g
		+ (unsigned __int128)t;
	d1 = (unsigned long long)z;
	t = (unsigned long long)(z >> 64);
	z = (unsigned __int128)tu.v2 * (unsigned __int128)f
		+ (unsigned __int128)tv.v2 * (unsigned __int128)g
		+ (unsigned __int128)t;
	d2 = (unsigned long long)z;
	t = (unsigned long long)(z >> 64);
	z = (unsigned __int128)tu.v3 * (unsigned __int128)f
		+ (unsigned __int128)tv.v3 * (unsigned __int128)g
		+ (unsigned __int128)t;
	d3 = (unsigned long long)z;
	t = (unsigned long long)(z >> 64);

	/*
	 * Upper word t can be up to 63 bits.
	 */
	z = (unsigned __int128)t * (unsigned __int128)(2 * MQ);
	cc = _addcarry_u64(0, d0, (unsigned long long)z, &d0);
	cc = _addcarry_u64(cc, d1, (unsigned long long)(z >> 64), &d1);
	cc = _addcarry_u64(cc, d2, 0, &d2);
	cc = _addcarry_u64(cc, d3, 0, &d3);

	/*
	 * There could be a carry, but in that case, the folding won't
	 * propagate beyond the second limb.
	 */
	cc = _addcarry_u64(0, d0, -(unsigned long long)cc & (2 * MQ), &d0);
	(void)_addcarry_u64(cc, d1, 0, &d1);

	d->v0 = d0;
	d->v1 = d1;
	d->v2 = d2;
	d->v3 = d3;
}

/* ================================================================== */
/*
 * Assembly code for the inner loop (generic version, can run for up to
 * 62 iterations).
 *    rax   f0
 *    rbx   g0
 *    rcx   f1
 *    rdx   g1
 *    rsi   xa
 *    rdi   xb
 */
#define INV_INNER \
	/* \
	 * Copy old values into extra registers \
	 *    r10   f0 \
	 *    r11   g0 \
	 *    r12   f1 \
	 *    r13   g1 \
	 *    r14   xa \
	 *    r15   xb \
	 */ \
	"movq	%%rax, %%r10\n\t" \
	"movq	%%rbx, %%r11\n\t" \
	"movq	%%rcx, %%r12\n\t" \
	"movq	%%rdx, %%r13\n\t" \
	"movq	%%rsi, %%r14\n\t" \
	"movq	%%rdi, %%r15\n\t" \
 \
	/* Conditional swap if xa < xb */ \
	"cmpq	%%rdi, %%rsi\n\t" \
	"cmovb	%%r15, %%rsi\n\t" \
	"cmovb	%%r14, %%rdi\n\t" \
	"cmovb	%%r12, %%rax\n\t" \
	"cmovb	%%r10, %%rcx\n\t" \
	"cmovb	%%r13, %%rbx\n\t" \
	"cmovb	%%r11, %%rdx\n\t" \
 \
	/* Subtract xb from xa */ \
	"subq	%%rdi, %%rsi\n\t" \
	"subq	%%rcx, %%rax\n\t" \
	"subq	%%rdx, %%rbx\n\t" \
 \
	/* If xa was even, override the operations above */ \
	"testl	$1, %%r14d\n\t" \
	"cmovz	%%r10, %%rax\n\t" \
	"cmovz	%%r11, %%rbx\n\t" \
	"cmovz	%%r12, %%rcx\n\t" \
	"cmovz	%%r13, %%rdx\n\t" \
	"cmovz	%%r14, %%rsi\n\t" \
	"cmovz	%%r15, %%rdi\n\t" \
 \
	/* Now xa is even; apply shift. */ \
	"shrq	$1, %%rsi\n\t" \
	"addq	%%rcx, %%rcx\n\t" \
	"addq	%%rdx, %%rdx\n\t"

/*
 * Alternate assembly code for the inner loop. This one groups values
 * by pairs and is slightly faster, but it is good only for up to 31
 * iterations.
 *    rax   f0:g0  (f0 = low half, g0 = high half)
 *    rcx   f1:g1
 *    rdx   0x7FFFFFFF7FFFFFFF
 *    rsi   xa
 *    rdi   xb
 */
#define INV_INNER_FAST \
	/* \
	 * Copy old values into extra registers \
	 *    r10   f0:g0 \
	 *    r12   f1:g1 \
	 *    r14   xa \
	 *    r15   xb \
	 */ \
	"movq	%%rax, %%r10\n\t" \
	"movq	%%rcx, %%r12\n\t" \
	"movq	%%rsi, %%r14\n\t" \
	"movq	%%rdi, %%r15\n\t" \
 \
	/* Conditional swap if xa < xb */ \
	"cmpq	%%rdi, %%rsi\n\t" \
	"cmovb	%%r15, %%rsi\n\t" \
	"cmovb	%%r14, %%rdi\n\t" \
	"cmovb	%%r12, %%rax\n\t" \
	"cmovb	%%r10, %%rcx\n\t" \
 \
	/* Subtract xb from xa */ \
	"subq	%%rdi, %%rsi\n\t" \
	"subq	%%rcx, %%rax\n\t" \
	"addq	%%rdx, %%rax\n\t" \
 \
	/* If xa was even, override the operations above */ \
	"testl	$1, %%r14d\n\t" \
	"cmovz	%%r10, %%rax\n\t" \
	"cmovz	%%r12, %%rcx\n\t" \
	"cmovz	%%r14, %%rsi\n\t" \
	"cmovz	%%r15, %%rdi\n\t" \
 \
	/* Now xa is even; apply shift. */ \
	"shrq	$1, %%rsi\n\t" \
	"addq	%%rcx, %%rcx\n\t" \
	"subq	%%rdx, %%rcx\n\t"

/* ================================================================== */

/* see gf25519.h */
uint64_t
gf_inv(gf *d, const gf *y)
{
	gf a, b, u, v;
	unsigned long long f0, f1, g0, g1, xa, xb;
	unsigned long long nega, negb;
	int i;
#if !GF_MODPRIME
	unsigned long long r;
#endif

	/*
	 * Extended binary GCD:
	 *
	 *   a <- y
	 *   b <- q
	 *   u <- 1
	 *   v <- 0
	 *
	 * a and b are nonnnegative integers (in the 0..q range). u
	 * and v are integers modulo q.
	 *
	 * Invariants:
	 *    a = y*u mod q
	 *    b = y*v mod q
	 *    b is always odd
	 *
	 * At each step:
	 *    if a is even, then:
	 *        a <- a/2, u <- u/2 mod q
	 *    else:
	 *        if a < b:
	 *            (a, u, b, v) <- (b, v, a, u)
	 *        a <- (a-b)/2, u <- (u-v)/2 mod q
	 *
	 * At one point, value a reaches 0; it will then stay there
	 * (all subsequent steps will only keep a at zero). The value
	 * of b is then GCD(y, p), i.e. 1 if y is invertible; value v
	 * is then the inverse of y modulo p.
	 *
	 * If y = 0, then a is initially at zero and stays there, and
	 * v is unchanged. The function then returns 0 as "inverse" of
	 * zero, which is the desired behaviour; therefore, no corrective
	 * action is necessary.
	 *
	 * It can be seen that each step will decrease the size of one of
	 * a and b by at least 1 bit; thus, starting with two values of
	 * at most 255 bits each, 509 iterations are always sufficient.
	 *
	 *
	 * In practice, we optimize this code in the following way:
	 *  - We do iterations by group of 31.
	 *  - In each group, we use _approximations_ of a and b that
	 *    fit on 64 bits each:
	 *      Let n = max(len(a), len(b)).
	 *      If n <= 64, then xa = na and xb = nb.
	 *      Otherwise:
	 *         xa = (a mod 2^31) + 2^31*floor(a / 2^(n - 33))
	 *         xb = (b mod 2^31) + 2^31*floor(b / 2^(n - 33))
	 *    I.e. we keep the same low 31 bits, but we remove all the
	 *    "middle" bits to just keep the higher bits. At least one
	 *    of the two values xa and xb will have maximum length (64 bits)
	 *    if either of a or b exceeds 64 bits.
	 *  - We record which subtract and swap we did into update
	 *    factors, which we apply en masse to a, b, u and v after
	 *    the 31 iterations.
	 *
	 * Since we kept the correct low 31 bits, all the "subtract or
	 * not" decisions are correct, but the "swap or not swap" might
	 * be wrong, since these use the difference between the two
	 * values, and we only have approximations thereof. A consequence
	 * is that after the update, a and/or b may be negative. If a < 0,
	 * we negate it, and also negate u (which we do by negating the
	 * update factors f0 and g0 before applying them to compute the
	 * new value of u); similary for b and v.
	 *
	 * It can be shown that the 31 iterations, along with the
	 * conditional negation, ensure that len(a)+len(b) is still
	 * reduced by at least 31 bits (compared with the classic binary
	 * GCD, the use of the approximations may make the code "miss one
	 * bit", but the conditional subtraction regains it). Thus,
	 * 509 iterations are sufficient in total. As explained later on,
	 * we can skip the final iteration as well.
	 */

	gf_normalize(&a, y);
	b = GF_P;
	u = GF_ONE;
	v = GF_ZERO;

	/*
	 * Generic loop first does 15*31 = 465 iterations.
	 */
	for (i = 0; i < 15; i ++) {
		unsigned long long m1, m2, m3, tnz1, tnz2, tnz3;
		unsigned long long tnzm, tnza, tnzb, snza, snzb;
		unsigned long long s, sm;
		gf na, nb, nu, nv;

		/*
		 * Get approximations of a and b over 64 bits:
		 *  - If len(a) <= 64 and len(b) <= 64, then we just
		 *    use the value (low limb).
		 *  - Otherwise, with n = max(len(a), len(b)), we use:
		 *       (a mod 2^31) + 2^31*(floor(a / 2^(n-33)))
		 *       (b mod 2^31) + 2^31*(floor(b / 2^(n-33)))
		 * I.e. we remove the "middle bits".
		 */
		m3 = a.v3 | b.v3;
		m2 = a.v2 | b.v2;
		m1 = a.v1 | b.v1;
		tnz3 = -((m3 | -m3) >> 63);
		tnz2 = -((m2 | -m2) >> 63) & ~tnz3;
		tnz1 = -((m1 | -m1) >> 63) & ~tnz3 & ~tnz2;
		tnzm = (m3 & tnz3) | (m2 & tnz2) | (m1 & tnz1);
		tnza = (a.v3 & tnz3) | (a.v2 & tnz2) | (a.v1 & tnz1);
		tnzb = (b.v3 & tnz3) | (b.v2 & tnz2) | (b.v1 & tnz1);
		snza = (a.v2 & tnz3) | (a.v1 & tnz2) | (a.v0 & tnz1);
		snzb = (b.v2 & tnz3) | (b.v1 & tnz2) | (b.v0 & tnz1);

		/*
		 * If both len(a) <= 64 and len(b) <= 64, then:
		 *    tnzm = 0
		 *    tnza = 0, snza = 0, tnzb = 0, snzb = 0
		 *    tnzm = 0
		 * Otherwise:
		 *    tnzm != 0, length yields value of n
		 *    tnza contains the top limb of a, snza the second limb
		 *    tnzb contains the top limb of b, snzb the second limb
		 *
		 * We count the number of leading zero bits in tnzm:
		 *  - If s <= 31, then the top 31 bits can be extracted
		 *    from tnza and tnzb alone.
		 *  - If 32 <= s <= 63, then we need some bits from snza
		 *    as well.
		 *
		 * We rely on the fact shifts don't reveal the shift count
		 * through side channels. This would not have been true on
		 * the Pentium IV, but it is true on all known x86 CPU that
		 * have 64-bit support and implement the LZCNT opcode.
		 */
		s = _lzcnt_u64(tnzm);
		sm = -((unsigned long long)(31 - s) >> 63);
		tnza ^= sm & (tnza ^ ((tnza << 32) | (snza >> 32)));
		tnzb ^= sm & (tnzb ^ ((tnzb << 32) | (snzb >> 32)));
		s -= 32 & sm;
		tnza <<= s;
		tnzb <<= s;

		/*
		 * At this point:
		 *  - If len(a) <= 64 and len(b) <= 64, then:
		 *       tnza = 0
		 *       tnzb = 0
		 *       tnz1 = tnz2 = tnz3 = 0
		 *  - Otherwise, we need to use the top 33 bits of tnza
		 *    and tnzb in combination with the low 31 bits of
		 *    a.v0 and b.v0, respectively.
		 */
		tnza |= a.v0 & ~(tnz1 | tnz2 | tnz3);
		tnzb |= b.v0 & ~(tnz1 | tnz2 | tnz3);
		xa = (a.v0 & 0x7FFFFFFF) | (tnza & 0xFFFFFFFF80000000);
		xb = (b.v0 & 0x7FFFFFFF) | (tnzb & 0xFFFFFFFF80000000);

		/*
		 * We can now run the binary GCD on xa and xb for 31
		 * rounds. We unroll it a bit (two rounds per loop
		 * iteration), it seems to save about 250 cycles in
		 * total on a Coffee Lake core.
		 */

		__asm__ __volatile__ (
			/*
			 * f0 = 1
			 * g0 = 0
			 * f1 = 0
			 * g1 = 1
			 * We add 0x7FFFFFFF to all four values, and
			 * group them by pairs into registers.
			 */
			"movq	$0x7FFFFFFF7FFFFFFF, %%rdx\n\t"
			"movq	$0x7FFFFFFF80000000, %%rax\n\t"
			"movq	$0x800000007FFFFFFF, %%rcx\n\t"

			/*
			 * Do the loop. Tests on a Coffee Lake core seem
			 * to indicate that not unrolling is best here.
			 * Loop counter is in r8.
			 */
			"movl	$31, %%r8d\n\t"
			"0:\n\t"
			INV_INNER_FAST
			"decl	%%r8d\n\t"
			"jnz	0b\n\t"

			/*
			 * Split f0, f1, g0 and g1 into separate variables.
			 */
			"movq	%%rax, %%rbx\n\t"
			"movq	%%rcx, %%rdx\n\t"
			"shrq	$32, %%rbx\n\t"
			"shrq	$32, %%rdx\n\t"
			"orl	%%eax, %%eax\n\t"
			"orl	%%ecx, %%ecx\n\t"
			"subq	$0x7FFFFFFF, %%rax\n\t"
			"subq	$0x7FFFFFFF, %%rbx\n\t"
			"subq	$0x7FFFFFFF, %%rcx\n\t"
			"subq	$0x7FFFFFFF, %%rdx\n\t"

			: "=a" (f0), "=b" (g0), "=c" (f1), "=d" (g1),
			  "=S" (xa), "=D" (xb)
			: "4" (xa), "5" (xb)
			: "cc", "r8", "r10", "r11",
			  "r12", "r13", "r14", "r15" );

		/*
		 * We now need to propagate updates to a, b, u and v.
		 */
		nega = s256_lin_div31_abs(&na, &a, &b, f0, g0);
		negb = s256_lin_div31_abs(&nb, &a, &b, f1, g1);
		f0 = (f0 ^ -nega) + nega;
		g0 = (g0 ^ -nega) + nega;
		f1 = (f1 ^ -negb) + negb;
		g1 = (g1 ^ -negb) + negb;
		gf_lin(&nu, &u, &v, f0, g0);
		gf_lin(&nv, &u, &v, f1, g1);
		a = na;
		b = nb;
		u = nu;
		v = nv;
	}

	/*
	 * At that point, if y is invertible, then the final GCD is 1,
	 * and len(a) + len(b) <= 45, so it is known that the values
	 * fully fit in a single register each. We can do the remaining
	 * 44 iterations in one go (they are exact, no approximation
	 * here). In fact, we can content ourselves with 43 iterations,
	 * because when arriving at the last iteration, we know that a = 0
	 * or 1 and b = 1 (the final state of the algorithm is that a = 0
	 * and b is the GCD of y and q), so there would be no swap. Since
	 * we only care about the update factors f1 and g1, we can simply
	 * avoid the final iteration.
	 *
	 * The update values f1 and g1, for v, will be up to 2^43 (in
	 * absolute value) but this is supported by gf_lin().
	 *
	 * If y is not invertible, then b does not necessarily fit in a
	 * single word. Thus, an extra verification step with a
	 * multiplication is performed, to check that the inverse is
	 * really an inverse. This step is not needed if the modulus p
	 * is prime, because, in that case, a non-invertible y can only
	 * be zero, in which case a = 0 and v = 0 throughout, which is
	 * the expected result.
	 */
	xa = a.v0;
	xb = b.v0;

	__asm__ __volatile__ (
		/* Set f0, g0, f1 and g1. */
		"movl	$1, %%eax\n\t"
		"xorl	%%ebx, %%ebx\n\t"
		"xorl	%%ecx, %%ecx\n\t"
		"movl	$1, %%edx\n\t"

		/* Do 43 iterations. We need to use the generic code
		   with one update factor per register, since we do
		   more than 31 iterations. Unrolling two iterations
		   in the loop appears to save a few cycles. */
		"movl	$21, %%r8d\n\t"
		"0:\n\t"
		INV_INNER
		INV_INNER
		"decl	%%r8d\n\t"
		"jnz	0b\n\t"
		INV_INNER

		: "=a" (f0), "=b" (g0), "=c" (f1), "=d" (g1),
		  "=S" (xa), "=D" (xb)
		: "4" (xa), "5" (xb)
		: "cc", "r8", "r10", "r11",
		  "r12", "r13", "r14", "r15" );

	gf_lin(&v, &u, &v, f1, g1);

	/*
	 * We did 31*15+43 = 508 iterations, and each injected a factor 2,
	 * thus we must divide by 2^508 (mod q).
	 */
#if GF_MODPRIME
	/*
	 * Result is correct if source operand was invertible, i.e.
	 * distinct from zero (since all non-zero values are invertible
	 * modulo a prime integer); the inverse is then also non-zero.
	 * If the source was zero, then the result is zero as well. We
	 * can thus test d instead of a.
	 */
	gf_mul_inline(d, &v, &GF_INVT508);
	return gf_iszero(d) ^ 1;
#else
	/*
	 * Verification of the computed inverse: for a non-prime modulus,
	 * there are non-invertible values other than zero, and the code
	 * above, especially with the optimizations in the final rounds,
	 * may have misdetected that case.
	 */
	gf_mul_inline(&v, &v, &GF_INVT508);
	gf_mul_inline(&u, &v, y);
	r = -gf_eq(&u, &GF_ONE);
	d->v0 = v.v0 & r;
	d->v1 = v.v1 & r;
	d->v2 = v.v2 & r;
	d->v3 = v.v3 & r;
	return r & 1;
#endif
}

static unsigned long long
dec64le(const uint8_t *buf)
{
	return (unsigned long long)buf[0]
		| ((unsigned long long)buf[1] << 8)
		| ((unsigned long long)buf[2] << 16)
		| ((unsigned long long)buf[3] << 24)
		| ((unsigned long long)buf[4] << 32)
		| ((unsigned long long)buf[5] << 40)
		| ((unsigned long long)buf[6] << 48)
		| ((unsigned long long)buf[7] << 56);
}

static void
enc64le(uint8_t *buf, unsigned long long x)
{
	buf[0] = (uint8_t)x;
	buf[1] = (uint8_t)(x >> 8);
	buf[2] = (uint8_t)(x >> 16);
	buf[3] = (uint8_t)(x >> 24);
	buf[4] = (uint8_t)(x >> 32);
	buf[5] = (uint8_t)(x >> 40);
	buf[6] = (uint8_t)(x >> 48);
	buf[7] = (uint8_t)(x >> 56);
}

/* see gf25519.h */
uint64_t
gf_decode(gf *d, const void *src)
{
	const uint8_t *buf;
	unsigned long long t;
	unsigned char cc;

	buf = src;
	d->v0 = dec64le(buf);
	d->v1 = dec64le(buf + 8);
	d->v2 = dec64le(buf + 16);
	d->v3 = dec64le(buf + 24);
	cc = _subborrow_u64(0, d->v0, GF_P.v0, &t);
	cc = _subborrow_u64(cc, d->v1, GF_P.v1, &t);
	cc = _subborrow_u64(cc, d->v2, GF_P.v2, &t);
	cc = _subborrow_u64(cc, d->v3, GF_P.v3, &t);
	return (uint64_t)cc;
}

/* see gf25519.h */
void
gf_encode(void *dst, const gf *a)
{
	gf t;
	uint8_t *buf;

	gf_normalize(&t, a);
	buf = dst;
	enc64le(buf, t.v0);
	enc64le(buf + 8, t.v1);
	enc64le(buf + 16, t.v2);
	enc64le(buf + 24, t.v3);
}
