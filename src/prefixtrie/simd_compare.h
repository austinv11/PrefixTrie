#ifndef SIMD_COMPARE_H
#define SIMD_COMPARE_H

#include <stddef.h>
#include <string.h>

// SIMD-optimized string comparison header
// This file provides hardware-accelerated string comparison functions
// that can process multiple bytes simultaneously for improved performance

#if defined(__AVX2__)
#include <immintrin.h>

/**
 * AVX2-optimized string comparison function
 * Compares strings 32 bytes at a time using 256-bit SIMD registers
 *
 * @param s1 First string to compare
 * @param s2 Second string to compare
 * @param n Maximum number of bytes to compare
 * @return Negative if s1 < s2, positive if s1 > s2, zero if equal
 */
static inline int simd_strncmp(const char* s1, const char* s2, size_t n) {
    size_t i = 0;

    // Process 32 bytes at a time using AVX2 instructions
    while (i + 32 <= n) {
        // Load 32 bytes from each string into 256-bit registers
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(s1 + i));
        __m256i v2 = _mm256_loadu_si256((const __m256i*)(s2 + i));

        // Compare all 32 bytes simultaneously
        // _mm256_cmpeq_epi8 returns 0xFF for equal bytes, 0x00 for different bytes
        // _mm256_movemask_epi8 extracts the high bit of each byte into a 32-bit mask
        int mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v2));

        // If mask != 0xFFFFFFFF, at least one byte differs
        if (mask != 0xFFFFFFFF) {
            // Find the first differing byte using bit manipulation
            #if defined(__GNUC__) || defined(__clang__)
            // Count trailing zeros in the inverted mask to find first difference
            int first_diff_byte_index = __builtin_ctz(~mask);
            #else
            // MSVC equivalent: bit scan forward for first set bit
            unsigned long first_diff_byte_index;
            _BitScanForward(&first_diff_byte_index, ~mask);
            #endif
            // Return the difference between the first differing bytes
            // Cast to unsigned char to handle signed char issues
            return (unsigned char)s1[i + first_diff_byte_index] - (unsigned char)s2[i + first_diff_byte_index];
        }
        i += 32;
    }

    // Handle remaining bytes (< 32) using standard library function
    return strncmp(s1 + i, s2 + i, n - i);
}

#elif defined(__SSE2__)
#include <emmintrin.h>

/**
 * SSE2-optimized string comparison function
 * Compares strings 16 bytes at a time using 128-bit SIMD registers
 *
 * @param s1 First string to compare
 * @param s2 Second string to compare
 * @param n Maximum number of bytes to compare
 * @return Negative if s1 < s2, positive if s1 > s2, zero if equal
 */
static inline int simd_strncmp(const char* s1, const char* s2, size_t n) {
    size_t i = 0;

    // Process 16 bytes at a time using SSE2 instructions
    while (i + 16 <= n) {
        // Load 16 bytes from each string into 128-bit registers
        __m128i v1 = _mm_loadu_si128((const __m128i*)(s1 + i));
        __m128i v2 = _mm_loadu_si128((const __m128i*)(s2 + i));

        // Compare all 16 bytes simultaneously
        // _mm_cmpeq_epi8 returns 0xFF for equal bytes, 0x00 for different bytes
        // _mm_movemask_epi8 extracts the high bit of each byte into a 16-bit mask
        int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(v1, v2));

        // If mask != 0xFFFF, at least one byte differs
        if (mask != 0xFFFF) {
            // Find the first differing byte using bit manipulation
            #if defined(__GNUC__) || defined(__clang__)
            // Count trailing zeros in the inverted mask to find first difference
            int first_diff_byte_index = __builtin_ctz(~mask);
            #else
            // MSVC equivalent: bit scan forward for first set bit
            unsigned long first_diff_byte_index;
            _BitScanForward(&first_diff_byte_index, ~mask);
            #endif
            // Return the difference between the first differing bytes
            // Cast to unsigned char to handle signed char issues
            return (unsigned char)s1[i + first_diff_byte_index] - (unsigned char)s2[i + first_diff_byte_index];
        }
        i += 16;
    }

    // Handle remaining bytes (< 16) using standard library function
    return strncmp(s1 + i, s2 + i, n - i);
}

#else
/**
 * Fallback implementation for platforms without SIMD support
 * Simply delegates to the standard library strncmp function
 *
 * @param s1 First string to compare
 * @param s2 Second string to compare
 * @param n Maximum number of bytes to compare
 * @return Negative if s1 < s2, positive if s1 > s2, zero if equal
 */
static inline int simd_strncmp(const char* s1, const char* s2, size_t n) {
    return strncmp(s1, s2, n);
}
#endif

#endif // SIMD_COMPARE_H
