package gf65536

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"testing"
)

// refMul and refMulAdd are the table-free oracle: one shift-and-reduce Mul per
// word, nothing shared with the kernels or with buildSplitTables.
func refMul(dst, src []byte, c uint16) {
	for w := 0; w+1 < len(src); w += 2 {
		p := Mul(c, uint16(src[w])<<8|uint16(src[w+1]))
		dst[w], dst[w+1] = byte(p>>8), byte(p)
	}
}

func refMulAdd(dst, src []byte, c uint16) {
	for w := 0; w+1 < len(src); w += 2 {
		p := Mul(c, uint16(src[w])<<8|uint16(src[w+1]))
		dst[w] ^= byte(p >> 8)
		dst[w+1] ^= byte(p)
	}
}

func allTiers() []tier { return tiers }

// withTier makes ti the implementation in use until the test ends.
func withTier(t *testing.T, ti tier) {
	t.Helper()
	old := cur
	cur = ti
	t.Cleanup(func() { cur = old })
}

func eachTier(t *testing.T, f func(t *testing.T)) {
	for _, ti := range allTiers() {
		t.Run(ti.name, func(t *testing.T) {
			withTier(t, ti)
			f(t)
		})
	}
}

func TestWantTier(t *testing.T) {
	want := os.Getenv("GF65536_WANT_TIER")
	if want == "" {
		t.Skip("GF65536_WANT_TIER not set")
	}
	if Tier() != want {
		t.Fatalf("tier %q in use, want %q (tiers: %d)", Tier(), want, len(tiers))
	}
	t.Logf("tier %s", Tier())
}

func TestMulField(t *testing.T) {
	// x^16 = x^12 + x^3 + x + 1 under the PAR2 polynomial.
	if got := Mul(0x8000, 2); got != 0x100B {
		t.Fatalf("Mul(0x8000, 2) = %#x, want 0x100b", got)
	}
	r := rand.New(rand.NewSource(11))
	for i := 0; i < 20000; i++ {
		a, b, c := uint16(r.Intn(1<<16)), uint16(r.Intn(1<<16)), uint16(r.Intn(1<<16))
		if Mul(a, b) != Mul(b, a) || Mul(a, 1) != a || Mul(a, 0) != 0 {
			t.Fatalf("identity/commutativity fails at %#x %#x", a, b)
		}
		if Mul(a, b^c) != Mul(a, b)^Mul(a, c) || Mul(Mul(a, b), c) != Mul(a, Mul(b, c)) {
			t.Fatalf("distributivity/associativity fails at %#x %#x %#x", a, b, c)
		}
	}
	// 2 generates the whole multiplicative group: 2^65535 = 1, no smaller power.
	x, order := uint16(2), 1
	for x != 1 {
		x = Mul(x, 2)
		order++
	}
	if order != 65535 {
		t.Fatalf("order of 2 is %d, want 65535", order)
	}
}

// Every constant: the split tables hold exactly (n<<4k)*c.
func TestSplitTablesAllConstants(t *testing.T) {
	var tbl [128]byte
	for c := 0; c < 1<<16; c++ {
		buildSplitTables(uint16(c), &tbl)
		for k := 0; k < 4; k++ {
			for n := 0; n < 16; n++ {
				p := Mul(uint16(c), uint16(n)<<(4*k))
				if tbl[32*k+n] != byte(p) || tbl[32*k+16+n] != byte(p>>8) {
					t.Fatalf("c=%#x k=%d n=%d: tables %#x%02x, want %#04x", c, k, n, tbl[32*k+16+n], tbl[32*k+n], p)
				}
			}
		}
	}
}

func coefficients() []uint16 {
	cs := []uint16{0, 1, 2, 3, 0x00FF, 0x0100, 0x100B, 0x1234, 0x7FFF, 0x8000, 0x8001, 0xABCD, 0xFF00, 0xFFFE, 0xFFFF}
	r := rand.New(rand.NewSource(5))
	for i := 0; i < 24; i++ {
		cs = append(cs, uint16(r.Intn(1<<16)))
	}
	return cs
}

func check(t *testing.T, dst, src []byte, c uint16) {
	t.Helper()
	got := append([]byte(nil), dst...)
	want := append([]byte(nil), dst...)
	MulAddSlice(got, src, c)
	refMulAdd(want, src, c)
	if !bytes.Equal(got, want) {
		t.Fatalf("MulAddSlice len=%d c=%#x differs from the reference", len(src), c)
	}
	MulSlice(got, src, c)
	refMul(want, src, c)
	if !bytes.Equal(got, want) {
		t.Fatalf("MulSlice len=%d c=%#x differs from the reference", len(src), c)
	}
}

// Lengths 0..1000 (every block boundary and tail), sampled and edge constants.
func TestLengths(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(1))
		cs := coefficients()
		for n := 0; n <= 1000; n++ {
			src := make([]byte, n)
			dst := make([]byte, n)
			r.Read(src)
			r.Read(dst)
			for _, c := range cs {
				check(t, dst, src, c)
			}
		}
	})
}

// Every one of the 65536 constants over a region that has two AVX2 blocks, one
// SSSE3-sized leftover block and a word tail.
func TestAllConstants(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(2))
		src := make([]byte, 64*2+32+2)
		dst := make([]byte, len(src))
		r.Read(src)
		r.Read(dst)
		for c := 0; c < 1<<16; c++ {
			check(t, dst, src, uint16(c))
		}
	})
}

func TestLargeAndUnaligned(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(3))
		buf := make([]byte, 1<<20+64)
		r.Read(buf)
		for off := 0; off < 3; off++ {
			src := buf[off : off+1<<20]
			dst := make([]byte, len(src)+3)[off : off+1<<20]
			r.Read(dst)
			check(t, dst, src, 0x9A3F)
		}
	})
}

func TestOddTailIgnored(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		src := bytes.Repeat([]byte{0xA5}, 67)
		dst := bytes.Repeat([]byte{0x3C}, 67)
		MulSlice(dst, src, 0x1234)
		if dst[66] != 0x3C {
			t.Fatalf("odd trailing byte written: %#x", dst[66])
		}
		want := bytes.Repeat([]byte{0x3C}, 67)
		refMul(want, src[:66], 0x1234)
		if !bytes.Equal(dst, want) {
			t.Fatal("even prefix differs from the reference")
		}
	})
}

// dst == src (exact overlap) is allowed.
func TestInPlace(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(4))
		for _, n := range []int{0, 2, 32, 64, 96, 130, 4096} {
			src := make([]byte, n)
			r.Read(src)
			want := make([]byte, n)
			refMul(want, src, 0xBEEF)
			MulSlice(src, src, 0xBEEF)
			if !bytes.Equal(src, want) {
				t.Fatalf("in-place MulSlice n=%d", n)
			}
		}
	})
}

func TestShortDstPanics(t *testing.T) {
	for _, f := range []func(dst, src []byte, c uint16){MulSlice, MulAddSlice} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("no panic for dst shorter than src")
				}
			}()
			f(make([]byte, 30), make([]byte, 32), 7)
		}()
	}
}

func FuzzMul(f *testing.F) {
	f.Add([]byte{0x12, 0x34, 0xAB, 0xCD}, uint16(7))
	f.Add(make([]byte, 64), uint16(0xFFFF))
	f.Add(make([]byte, 66), uint16(0))
	f.Add(bytes.Repeat([]byte{0xFF}, 130), uint16(1))
	f.Add(bytes.Repeat([]byte{0x81}, 224), uint16(0x8000))
	f.Fuzz(func(t *testing.T, data []byte, c uint16) {
		dst := make([]byte, len(data))
		for i := range dst {
			dst[i] = data[len(data)-1-i]
		}
		for _, ti := range allTiers() {
			old := cur
			cur = ti
			check(t, dst, data, c)
			cur = old
		}
	})
}

func BenchmarkMulAdd(b *testing.B) {
	for _, ti := range allTiers() {
		for _, size := range []int{1 << 10, 1 << 16, 1 << 20} {
			b.Run(fmt.Sprintf("%s/%d", ti.name, size), func(b *testing.B) {
				old := cur
				cur = ti
				defer func() { cur = old }()
				src := make([]byte, size)
				rand.New(rand.NewSource(3)).Read(src)
				dst := make([]byte, size)
				b.SetBytes(int64(size))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MulAddSlice(dst, src, 0x9A3F)
				}
			})
		}
	}
}

func BenchmarkMul(b *testing.B) {
	for _, ti := range allTiers() {
		b.Run(ti.name, func(b *testing.B) {
			old := cur
			cur = ti
			defer func() { cur = old }()
			src := make([]byte, 1<<20)
			rand.New(rand.NewSource(3)).Read(src)
			dst := make([]byte, len(src))
			b.SetBytes(int64(len(src)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MulSlice(dst, src, 0x9A3F)
			}
		})
	}
}
