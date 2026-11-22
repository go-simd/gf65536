//go:build !amd64 && !arm64 && !s390x && !loong64 && !ppc64le && !riscv64

package gf65536

// No SIMD kernel on this architecture: everything runs in the scalar tier.
func archTiers() []tier { return nil }
