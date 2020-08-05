#ifndef GF25519_H__
#define GF25519_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Type for a field element.
 */
typedef struct {
	uint64_t v0, v1, v2, v3;
} gf;

/*
 * Decode a field element from 32 bytes (little-endian convention). All
 * 256 bits are used. Returned value is 1 if the provided element was in
 * the 0..p-1 range (canonical encoding), 0 otherwise. If 0 is returned,
 * then a reduction modulo p was implicitly applied.
 */
uint64_t gf_decode(gf *d, const void *src);

/*
 * Encode a field element into 32 bytes (little-endian convention). This
 * always produces a canonical encoding, i.e. in the 0..p-1 range.
 */
void gf_encode(void *dst, const gf *a);

/*
 * Operations in the finite field; 'd' is the destination. In all
 * functions, the destination structure may be the same as one or both
 * of the sources.
 *
 * gf_condneg() performs a conditional negation: d is set to -a if
 * ctl == 1; d is left unchanged if ctl == 0. ctl MUST be 0 or 1.
 *
 * gf_sqr_x() performs 'num' successive squarings. The parameter is
 * internally truncated to 32 bits (which is already a lot more than
 * makes sense).
 *
 * gf_inv() returns 1 if the input is invertible, 0 otherwise. In the
 * latter case, d is filled with the value zero.
 */
void gf_add(gf *d, const gf *a, const gf *b);
void gf_sub(gf *d, const gf *a, const gf *b);
void gf_neg(gf *d, const gf *a);
void gf_condneg(gf *d, const gf *a, uint64_t ctl);
void gf_mul(gf *d, const gf *a, const gf *b);
void gf_sqr(gf *d, const gf *a);
void gf_sqr_x(gf *d, const gf *a, long num);
uint64_t gf_inv(gf *d, const gf *a);

/*
 * For performance comparisons only, if GF_INV_FLT is defined to
 * a non-zero value, then the gf_inv_FLT() function is defined and
 * implements inversion in the field with Fermat's little theorem.
 * Furthermore, if GF_INV_FLT is defined to 2 or more, then the code
 * will inline calls to multiplications and squarings, making it
 * larger but slightly faster as well.
 *
 * gf_inv() is substantially faster than gf_inv_FLT(), and thus
 * always preferable.
 */
#define GF_INV_FLT   2

#if defined GF_INV_FLT && GF_INV_FLT
/*
 * For comparisons only, gf_inv_FLT() computes inverses using
 * Fermat's little theorem; it is slower than gf_inv().
 */
uint64_t gf_inv_FLT(gf *d, const gf *a);
#endif

/*
 * Equality test: returned value is 1 on equality, 0 otherwise.
 * gf_iszero() compares with zero.
 */
uint64_t gf_eq(const gf *a, const gf *b);
uint64_t gf_iszero(const gf *a);

#ifdef __cplusplus
}
#endif

#endif
