// Package gf65536 multiplies whole regions of 16-bit words by a constant in
// GF(2^16) with SIMD split-table kernels.
//
// The field is GF(2)[x]/(x^16 + x^12 + x^3 + x + 1), polynomial 0x1100B, the
// field of the PAR2 recovery format, with generator 2. A region is a byte slice
// read as big-endian uint16 words (first byte is the high byte), the layout
// go-erasure/reedsolomon stores shards in.
//
// For a fixed constant c the product is GF(2)-linear in the word, so the four
// nibbles n0..n3 of a word v give c*v = T0[n0] ^ T1[n1] ^ T2[n2] ^ T3[n3], each
// Tk a 16-entry table of 16-bit products. Splitting every table into a low-byte
// and a high-byte table makes eight 16-entry byte tables, each one a byte
// shuffle (PSHUFB / TBL / VPERM / vrgather) on the byte planes of the region
// (J. S. Plank, K. M. Greenan, E. L. Miller, "Screaming Fast Galois Field
// Arithmetic Using Intel SIMD Instructions", FAST 2013). The kernels fold whole
// 32-byte blocks (64-byte on AVX2); the tail is finished with the same table in
// scalar code.
package gf65536

// Poly is the reduction polynomial x^16 + x^12 + x^3 + x + 1.
const Poly = 0x1100B

// blockBytes is the granularity of the 128-bit kernels: 32 bytes, 16 words.
const blockBytes = 32

// Mul returns a*b in GF(2^16). It is table-free (shift and reduce), meant for
// building tables and as the test oracle, not for bulk work: use MulSlice.
func Mul(a, b uint16) uint16 {
	var p uint32
	x, y := uint32(a), uint32(b)
	for y != 0 {
		if y&1 != 0 {
			p ^= x
		}
		y >>= 1
		x <<= 1
		if x&0x10000 != 0 {
			x ^= Poly
		}
	}
	return uint16(p)
}

// buildSplitTables fills tbl for the constant c: the 128-byte layout is
// T0L,T0H,T1L,T1H,T2L,T2H,T3L,T3H, each 16 bytes, where TkL[n] and TkH[n] are the
// low and high byte of (n<<4k)*c. The products are linear in the bits of n, so
// they come from the 16 values c*x^i by XOR instead of 64 multiplies.
func buildSplitTables(c uint16, tbl *[128]byte) {
	var basis [16]uint16
	b := uint32(c)
	for i := range basis {
		basis[i] = uint16(b)
		b <<= 1
		if b&0x10000 != 0 {
			b ^= Poly
		}
	}
	for k := 0; k < 4; k++ {
		for n := 0; n < 16; n++ {
			var p uint16
			for j := 0; j < 4; j++ {
				if n>>j&1 != 0 {
					p ^= basis[4*k+j]
				}
			}
			tbl[32*k+n] = byte(p)
			tbl[32*k+16+n] = byte(p >> 8)
		}
	}
}

// tier is one implementation of the region kernels. mul and mulAdd process a
// prefix of the regions and return its length in bytes (a multiple of 32, maybe
// 0); the caller finishes the rest in scalar code.
type tier struct {
	name   string
	mul    func(dst, src []byte, tbl *[128]byte) int
	mulAdd func(dst, src []byte, tbl *[128]byte) int
}

// blockTier wraps a pair of assembly kernels that fold n whole 32-byte blocks.
func blockTier(name string, mul, mulAdd func(dst, src []byte, tbl *[128]byte, n int)) tier {
	return tier{name,
		func(dst, src []byte, tbl *[128]byte) int {
			n := len(src) &^ (blockBytes - 1)
			if n > 0 {
				mul(dst[:n], src[:n], tbl, n/blockBytes)
			}
			return n
		},
		func(dst, src []byte, tbl *[128]byte) int {
			n := len(src) &^ (blockBytes - 1)
			if n > 0 {
				mulAdd(dst[:n], src[:n], tbl, n/blockBytes)
			}
			return n
		}}
}

var scalarTier = tier{"scalar",
	func(dst, src []byte, tbl *[128]byte) int { return 0 },
	func(dst, src []byte, tbl *[128]byte) int { return 0 },
}

// tiers lists the usable implementations, fastest first, ending with the scalar
// one; cur is the one in use (tests switch it).
var (
	tiers = append(archTiers(), scalarTier)
	cur   = tiers[0]
)

// Tier names the implementation MulSlice and MulAddSlice use on this machine:
// "avx2", "ssse3", "neon", "lsx", "vsx", "vx", "rvv" or "scalar".
func Tier() string { return cur.name }

// MulSlice sets dst = src*c for every 16-bit big-endian word of src. dst and src
// must be the same slice or disjoint; len(dst) must be at least len(src). An odd
// trailing byte of src is ignored (dst is left untouched there). It panics if
// dst is shorter than src.
func MulSlice(dst, src []byte, c uint16) {
	n := len(src) &^ 1
	if len(dst) < n {
		panic("gf65536: dst shorter than src")
	}
	dst, src = dst[:n], src[:n]
	switch c {
	case 0:
		clear(dst)
		return
	case 1:
		copy(dst, src)
		return
	}
	var tbl [128]byte
	buildSplitTables(c, &tbl)
	k := cur.mul(dst, src, &tbl)
	mulTail(dst[k:], src[k:], &tbl)
}

// MulAddSlice sets dst ^= src*c for every 16-bit big-endian word of src, with
// the same preconditions as MulSlice.
func MulAddSlice(dst, src []byte, c uint16) {
	n := len(src) &^ 1
	if len(dst) < n {
		panic("gf65536: dst shorter than src")
	}
	dst, src = dst[:n], src[:n]
	switch c {
	case 0:
		return
	case 1:
		for i := range src {
			dst[i] ^= src[i]
		}
		return
	}
	var tbl [128]byte
	buildSplitTables(c, &tbl)
	k := cur.mulAdd(dst, src, &tbl)
	mulAddTail(dst[k:], src[k:], &tbl)
}

// product returns the word v*c from the tables, high byte first.
func product(tbl *[128]byte, hi, lo byte) (phi, plo byte) {
	n0, n1, n2, n3 := lo&15, lo>>4, hi&15, hi>>4
	plo = tbl[n0] ^ tbl[32+n1] ^ tbl[64+n2] ^ tbl[96+n3]
	phi = tbl[16+n0] ^ tbl[48+n1] ^ tbl[80+n2] ^ tbl[112+n3]
	return
}

func mulTail(dst, src []byte, tbl *[128]byte) {
	for w := 0; w+1 < len(src); w += 2 {
		dst[w], dst[w+1] = product(tbl, src[w], src[w+1])
	}
}

func mulAddTail(dst, src []byte, tbl *[128]byte) {
	for w := 0; w+1 < len(src); w += 2 {
		hi, lo := product(tbl, src[w], src[w+1])
		dst[w] ^= hi
		dst[w+1] ^= lo
	}
}
