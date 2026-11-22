# gf65536

Region multiply over GF(2^16) with SIMD split-table kernels, for Reed-Solomon
erasure codes and PAR2-style recovery.

```go
import "github.com/go-simd/gf65536"

gf65536.MulSlice(dst, src, c)    // dst  = src * c
gf65536.MulAddSlice(dst, src, c) // dst ^= src * c
```

- Field: GF(2)[x]/(x^16 + x^12 + x^3 + x + 1), polynomial `0x1100B`, generator 2:
  the field of PAR2 and of [go-erasure/reedsolomon](https://github.com/go-erasure/reedsolomon).
- A region is a byte slice read as **big-endian** `uint16` words (first byte is
  the high byte), the layout reedsolomon stores shards in. A little-endian
  variant (PAR2 files store words little-endian) is not provided yet.
- `dst` and `src` must be the same slice or disjoint, `len(dst) >= len(src)`
  (otherwise it panics); an odd trailing byte of `src` is ignored.
- Results are exact: every tier is tested against a table-free shift-and-reduce
  scalar reference, for all 65536 constants and for lengths 0..1000.
- `gf65536.Mul(a, b)` is that table-free reference; `gf65536.Tier()` names the
  kernel in use.

## Tiers

| Architecture | Kernel | Block |
|---|---|---|
| amd64 | AVX2 `VPSHUFB`, then SSSE3 `PSHUFB` | 64 B / 32 B |
| arm64 | NEON `VLD2` + `VTBL` | 32 B |
| loong64 | LSX `vshuf.b` | 32 B |
| ppc64le | VSX `VPERM` (POWER8 baseline) | 32 B |
| riscv64 | RVV `vrgather.vv` (needs V) | 32 B |
| s390x | vector facility `VPERM` | 32 B |
| others | scalar, same tables | |

For a constant `c`, `c*v = T0[n0]^T1[n1]^T2[n2]^T3[n3]` over the four nibbles of
`v`; each `Tk` is split into a low-byte and a high-byte 16-entry table, so a
kernel deinterleaves the region into a high and a low byte plane, does eight byte
shuffles and re-interleaves. The 128-byte table for `c` is built in about 64 XORs
per call. The kernels are generated with
[go-asmgen](https://github.com/go-asmgen/asmgen) (`gen/`); CI regenerates them and
fails on any difference.

## GFNI

Not used. `GF2P8AFFINEQB` could replace the eight shuffles by four affine maps
over the deinterleaved planes (each output byte is an XOR of two 8x8 GF(2) maps
of the two input bytes), but no GFNI hardware is available to us for
verification or measurement (hosted runners and QEMU lack it), and an
unmeasured kernel is not shipped. See the issue tracker.

## Testing

`go test ./...` runs every tier the machine offers (and the scalar tier) against
the reference: all constants, all lengths 0..1000, in-place, unaligned, guard pages
on both sides of both operands, and a fuzz target. CI runs native amd64 (AVX2,
and SSSE3 with AVX2 masked) and arm64, and riscv64, loong64, ppc64le (POWER9 and
POWER8) and s390x under QEMU.
