//go:build unix

package gf65536

import (
	"bytes"
	"testing"

	"golang.org/x/sys/unix"
)

// guarded returns two n-byte windows into one anonymous mapping laid out as
// [PROT_NONE][data...][PROT_NONE]; the first window ends flush against the
// trailing guard page and the second starts flush after the leading one, so an
// over-read or over-write by one byte on either side faults.
func guarded(t *testing.T, n int) (front, back []byte) {
	t.Helper()
	ps := unix.Getpagesize()
	size := (n + ps - 1) / ps * ps
	mem, err := unix.Mmap(-1, 0, size+2*ps, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_ANON|unix.MAP_PRIVATE)
	if err != nil {
		t.Skipf("mmap: %v", err)
	}
	t.Cleanup(func() { unix.Munmap(mem) })
	if err := unix.Mprotect(mem[:ps], unix.PROT_NONE); err != nil {
		t.Skipf("mprotect: %v", err)
	}
	if err := unix.Mprotect(mem[ps+size:], unix.PROT_NONE); err != nil {
		t.Skipf("mprotect: %v", err)
	}
	return mem[ps+size-n : ps+size], mem[ps : ps+n]
}

// A kernel that touches a byte outside [0,n) of either operand dies with
// SIGSEGV and takes the test binary with it, which is the failure we want.
func TestGuardPages(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		for n := 0; n <= 300; n += 2 {
			for _, flush := range []string{"end", "start"} {
				dstEnd, dstStart := guarded(t, n)
				srcEnd, srcStart := guarded(t, n)
				dst, src := dstEnd, srcEnd
				if flush == "start" {
					dst, src = dstStart, srcStart
				}
				for i := range src {
					src[i] = byte(i*13 + n)
					dst[i] = byte(i * 5)
				}
				want := append([]byte{}, dst...)
				refMulAdd(want, src, 0xC3A5)
				MulAddSlice(dst, src, 0xC3A5)
				if !bytes.Equal(dst, want) {
					t.Fatalf("MulAddSlice n=%d %s", n, flush)
				}
				refMul(want, src, 0xC3A5)
				MulSlice(dst, src, 0xC3A5)
				if !bytes.Equal(dst, want) {
					t.Fatalf("MulSlice n=%d %s", n, flush)
				}
			}
		}
	})
}
