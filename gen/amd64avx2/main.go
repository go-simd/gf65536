// Command gen produces galois_amd64_avx2.s with go-asmgen: the AVX2 split-table
// region multiply over GF(2^16). It emits galMulAVX2 (dst = src*coeff) and
// galMulAddAVX2 (dst ^= src*coeff), each folding n whole 64-byte blocks (32
// big-endian uint16 words) per call.
//
// It is the 256-bit form of the SSSE3 kernel (galois_amd64_gen.go) and uses the
// same 128-byte table (see buildSplitTables in gf65536.go), broadcast to both
// 128-bit lanes: VPSHUFB looks up inside each lane, so no cross-lane shuffle is
// ever needed. Per 64-byte block (two YMM loads A, B of interleaved hi,lo bytes):
//
//   - One VPSHUFB per load with the mask {0,2,..,14,1,3,..,15} turns each lane
//     into [even bytes | odd bytes]; VPUNPCKLQDQ / VPUNPCKHQDQ of A', B' then give
//     the HI plane [Ae|Be] and the LO plane [Ao|Bo] of each lane.
//   - Four nibble vectors (LO&0F, LO>>4&0F, HI&0F, HI>>4&0F), eight VPSHUFB
//     lookups, XORed into an output LO plane and HI plane.
//   - VPUNPCKLBW / VPUNPCKHBW of (HI, LO) restore, lane by lane, exactly the
//     layout of A and of B, so both are stored (or XORed into dst) in place.
//
// Registers: Y0..Y7 tables T0L,T0H,T1L,T1H,T2L,T2H,T3L,T3H; Y14 the shuffle mask;
// Y15 the 0x0F mask; Y8..Y13 working.
//
// Signature: galMulAVX2(dst, src []byte, tbl *[128]byte, n int), n = number of
// whole 64-byte blocks. Run: cd gen && go run galois_amd64_avx2_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{
			abi.Slice("dst"), abi.Slice("src"),
			abi.Scalar("tbl", abi.Ptr), abi.Scalar("n", abi.Int64),
		},
		nil,
	)
}

func rep(b byte) []byte {
	out := make([]byte, 16)
	for i := range out {
		out[i] = b
	}
	return out
}

// evenOdd gathers the even bytes into the low 8 lanes and the odd bytes into the
// high 8 lanes of a 16-byte vector.
func evenOdd() []byte {
	m := make([]byte, 16)
	for i := 0; i < 8; i++ {
		m[i] = byte(2 * i)
		m[8+i] = byte(2*i + 1)
	}
	return m
}

func body(b *amd64.Builder, muladd bool) {
	b.
		Raw("VMOVDQU 0(SI), Y8").Raw("VMOVDQU 32(SI), Y9").
		Raw("VPSHUFB Y14, Y8, Y8").Raw("VPSHUFB Y14, Y9, Y9").
		// Y10 = LO plane [Ao|Bo], Y11 = HI plane [Ae|Be].
		Raw("VPUNPCKHQDQ Y9, Y8, Y10").Raw("VPUNPCKLQDQ Y9, Y8, Y11").
		// n0 = Y12, n1 = Y10, n2 = Y13, n3 = Y11.
		Raw("VPAND Y15, Y10, Y12").
		Raw("VPSRLW $4, Y10, Y10").Raw("VPAND Y15, Y10, Y10").
		Raw("VPAND Y15, Y11, Y13").
		Raw("VPSRLW $4, Y11, Y11").Raw("VPAND Y15, Y11, Y11").
		// LO-out (Y8) and HI-out (Y9).
		Raw("VPSHUFB Y12, Y0, Y8").
		Raw("VPSHUFB Y12, Y1, Y9").
		Raw("VPSHUFB Y10, Y2, Y12").Raw("VPXOR Y12, Y8, Y8").
		Raw("VPSHUFB Y10, Y3, Y12").Raw("VPXOR Y12, Y9, Y9").
		Raw("VPSHUFB Y13, Y4, Y12").Raw("VPXOR Y12, Y8, Y8").
		Raw("VPSHUFB Y13, Y5, Y12").Raw("VPXOR Y12, Y9, Y9").
		Raw("VPSHUFB Y11, Y6, Y12").Raw("VPXOR Y12, Y8, Y8").
		Raw("VPSHUFB Y11, Y7, Y12").Raw("VPXOR Y12, Y9, Y9").
		// Re-interleave (HI first): Y10 = A's layout, Y11 = B's layout.
		Raw("VPUNPCKLBW Y8, Y9, Y10").Raw("VPUNPCKHBW Y8, Y9, Y11")
	if muladd {
		b.Raw("VPXOR 0(DX), Y10, Y10").Raw("VPXOR 32(DX), Y11, Y11")
	}
	b.Raw("VMOVDQU Y10, 0(DX)").Raw("VMOVDQU Y11, 32(DX)")
}

func gen(f *emit.File, name string, muladd bool, mask, lo string) {
	b := amd64.NewFunc(name, sig(), 0)
	b.LoadArg("dst_base", "DX").LoadArg("src_base", "SI").
		LoadArg("tbl", "DI").LoadArg("n", "CX").
		Raw("TESTQ CX, CX").Raw("JZ done")
	for i := 0; i < 8; i++ {
		b.Raw("VBROADCASTI128 %d(DI), Y%d", 16*i, i)
	}
	b.Raw("VBROADCASTI128 %s+0(SB), Y14", mask).
		Raw("VBROADCASTI128 %s+0(SB), Y15", lo).
		Label("loop")
	body(b, muladd)
	b.
		Raw("ADDQ $64, SI").Raw("ADDQ $64, DX").
		Raw("SUBQ $1, CX").Raw("JNZ loop").
		Label("done").Raw("VZEROUPPER").Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("amd64")
	mask := f.Data("evenOddMask", evenOdd())
	lo := f.Data("loNibY", rep(0x0f))
	gen(f, "galMulAVX2", false, mask, lo)
	gen(f, "galMulAddAVX2", true, mask, lo)
	if err := os.WriteFile("../../galois_amd64_avx2.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote galois_amd64_avx2.s")
}
