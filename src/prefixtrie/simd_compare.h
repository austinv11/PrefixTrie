#ifndef SIMD_COMPARE_H
#define SIMD_COMPARE_H

#include <stddef.h>
#include <string.h>

// SIMD-optimized string utility header.
// This file provides hardware-accelerated string functions using vector instructions
// to process multiple bytes simultaneously for improved performance.
// It is structured to define a function signature, then provide an
// implementation chosen by the compiler based on supported CPU features.

// -----------------------------
// Function Forward Declarations
// -----------------------------

/**
 * Compares up to n characters of the string s1 to those of the string s2.
 * This function is a SIMD-accelerated version of strncmp.
 * Uses vector instructions to compare 16-32 bytes at once instead of byte-by-byte.
 */
static inline int simd_strncmp(const char* s1, const char* s2, size_t n);

/**
 * Finds the first occurrence of any character from `needles` in `s`.
 * This function is a SIMD-accelerated equivalent of a multi-character strchr.
 * Uses vector instructions to check multiple characters against multiple needles simultaneously.
 * @return Pointer to the first match, or NULL if no match is found.
 */
static inline const char* simd_strchr_any(const char* s, size_t n, const char* needles, size_t num_needles);


// -----------------------------
// AVX2 Implementation (256-bit vectors, 32 bytes at a time)
// -----------------------------
#if defined(__AVX2__)
#include <immintrin.h>

static inline int simd_strncmp(const char* s1, const char* s2, size_t n) {
    size_t i = 0;

    // Process 32-byte chunks using AVX2 256-bit vector instructions
    while (i + 32 <= n) {
        // Load 32 bytes from each string into 256-bit vectors
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(s1 + i));
        __m256i v2 = _mm256_loadu_si256((const __m256i*)(s2 + i));

        // Compare all 32 bytes simultaneously, creating a mask where equal bytes = 0xFF
        __m256i equal_result = _mm256_cmpeq_epi8(v1, v2);

        // Extract comparison result as a 32-bit mask (one bit per byte)
        int mask = _mm256_movemask_epi8(equal_result);

        // If mask != 0xFFFFFFFF, at least one byte differs
        if (mask != 0xFFFFFFFF) {
            // Find the first differing byte using bit manipulation
            #if defined(__GNUC__) || defined(__clang__)
            // Count trailing zeros in inverted mask to find first difference
            int first_diff_byte_index = __builtin_ctz(~mask);
            #else
            // MSVC equivalent of counting trailing zeros
            unsigned long first_diff_byte_index;
            _BitScanForward(&first_diff_byte_index, ~mask);
            #endif
            // Return standard strcmp result: difference of first differing bytes
            return (unsigned char)s1[i + first_diff_byte_index] - (unsigned char)s2[i + first_diff_byte_index];
        }
        i += 32;  // Move to next 32-byte chunk
    }

    // Handle remaining bytes (< 32) using standard strncmp
    return strncmp(s1 + i, s2 + i, n - i);
}

static inline const char* simd_strchr_any(const char* s, size_t n, const char* needles, size_t num_needles) {
    size_t i = 0;

    // Process 32-byte chunks using AVX2 256-bit vector instructions
    while (i + 32 <= n) {
        // Load 32 bytes of haystack into vector
        __m256i v_haystack = _mm256_loadu_si256((const __m256i*)(s + i));

        // Initialize combined result mask to all zeros
        __m256i combined_mask = _mm256_setzero_si256();

        // Check each needle character against all 32 haystack bytes
        for (size_t j = 0; j < num_needles; ++j) {
            // Broadcast single needle character to all 32 positions in vector
            __m256i v_needle = _mm256_set1_epi8(needles[j]);

            // Compare needle against all haystack bytes simultaneously
            __m256i mask = _mm256_cmpeq_epi8(v_haystack, v_needle);

            // OR the result with combined mask to accumulate matches
            combined_mask = _mm256_or_si256(combined_mask, mask);
        }

        // Extract match results as 32-bit mask (one bit per byte)
        int mask = _mm256_movemask_epi8(combined_mask);

        // If any bit is set, we found a match
        if (mask != 0) {
            // Find position of first match using bit manipulation
            #if defined(__GNUC__) || defined(__clang__)
            // Count trailing zeros to find first set bit
            int first_match_index = __builtin_ctz(mask);
            #else
            // MSVC equivalent
            unsigned long first_match_index;
            _BitScanForward(&first_match_index, mask);
            #endif
            // Return pointer to first matching character
            return s + i + first_match_index;
        }
        i += 32;  // Move to next 32-byte chunk
    }

    // Handle remaining bytes (< 32) with simple loop
    for (; i < n; ++i) {
        for (size_t j = 0; j < num_needles; ++j) {
            if (s[i] == needles[j]) return s + i;
        }
    }
    return NULL;
}


// -----------------------------
// SSE2 Implementation (128-bit vectors, 16 bytes at a time)
// -----------------------------
#elif defined(__SSE2__)
#include <emmintrin.h>

static inline int simd_strncmp(const char* s1, const char* s2, size_t n) {
    size_t i = 0;

    // Process 16-byte chunks using SSE2 128-bit vector instructions
    while (i + 16 <= n) {
        // Load 16 bytes from each string into 128-bit vectors
        __m128i v1 = _mm_loadu_si128((const __m128i*)(s1 + i));
        __m128i v2 = _mm_loadu_si128((const __m128i*)(s2 + i));

        // Compare all 16 bytes simultaneously, creating mask where equal bytes = 0xFF
        __m128i equal_result = _mm_cmpeq_epi8(v1, v2);

        // Extract comparison result as a 16-bit mask (one bit per byte)
        int mask = _mm_movemask_epi8(equal_result);

        // If mask != 0xFFFF, at least one byte differs
        if (mask != 0xFFFF) {
            // Find the first differing byte
            #if defined(__GNUC__) || defined(__clang__)
            // Count trailing zeros in inverted mask to find first difference
            int first_diff_byte_index = __builtin_ctz(~mask);
            #else
            // MSVC equivalent
            unsigned long first_diff_byte_index;
            _BitScanForward(&first_diff_byte_index, ~mask);
            #endif
            // Return standard strcmp result: difference of first differing bytes
            return (unsigned char)s1[i + first_diff_byte_index] - (unsigned char)s2[i + first_diff_byte_index];
        }
        i += 16;  // Move to next 16-byte chunk
    }

    // Handle remaining bytes (< 16) using standard strncmp
    return strncmp(s1 + i, s2 + i, n - i);
}

static inline const char* simd_strchr_any(const char* s, size_t n, const char* needles, size_t num_needles) {
    size_t i = 0;

    // Process 16-byte chunks using SSE2 128-bit vector instructions
    while (i + 16 <= n) {
        // Load 16 bytes of haystack into vector
        __m128i v_haystack = _mm_loadu_si128((const __m128i*)(s + i));

        // Initialize combined result mask to all zeros
        __m128i combined_mask = _mm_setzero_si128();

        // Check each needle character against all 16 haystack bytes
        for (size_t j = 0; j < num_needles; ++j) {
            // Broadcast single needle character to all 16 positions in vector
            __m128i v_needle = _mm_set1_epi8(needles[j]);

            // Compare needle against all haystack bytes simultaneously
            __m128i mask = _mm_cmpeq_epi8(v_haystack, v_needle);

            // OR the result with combined mask to accumulate matches
            combined_mask = _mm_or_si128(combined_mask, mask);
        }

        // Extract match results as 16-bit mask (one bit per byte)
        int mask = _mm_movemask_epi8(combined_mask);

        // If any bit is set, we found a match
        if (mask != 0) {
            // Find position of first match
            #if defined(__GNUC__) || defined(__clang__)
            // Count trailing zeros to find first set bit
            int first_match_index = __builtin_ctz(mask);
            #else
            // MSVC equivalent
            unsigned long first_match_index;
            _BitScanForward(&first_match_index, mask);
            #endif
            // Return pointer to first matching character
            return s + i + first_match_index;
        }
        i += 16;  // Move to next 16-byte chunk
    }

    // Handle remaining bytes (< 16) with simple loop
    for (; i < n; ++i) {
        for (size_t j = 0; j < num_needles; ++j) {
            if (s[i] == needles[j]) return s + i;
        }
    }
    return NULL;
}


// -----------------------------
// ARM NEON Implementation (128-bit vectors, 16 bytes at a time)
// -----------------------------
#elif defined(__ARM_NEON)
#include <arm_neon.h>

static inline int simd_strncmp(const char* s1, const char* s2, size_t n) {
    size_t i = 0;

// Only use NEON intrinsics on 64-bit ARM with compiler support
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    // Process 16-byte chunks using NEON 128-bit vector instructions
    while (i + 16 <= n) {
        // Load 16 bytes from each string into 128-bit vectors
        uint8x16_t v1 = vld1q_u8((const uint8_t*)(s1 + i));
        uint8x16_t v2 = vld1q_u8((const uint8_t*)(s2 + i));

        // Compare all 16 bytes simultaneously, creating mask where equal bytes = 0xFF
        uint8x16_t equal_mask = vceqq_u8(v1, v2);

        // Convert to 64-bit view for easier bit manipulation
        uint64x2_t v_u64 = vreinterpretq_u64_u8(equal_mask);

        // Check if all bytes are equal (all bits set in both 64-bit lanes)
        if ((vgetq_lane_u64(v_u64, 0) & vgetq_lane_u64(v_u64, 1)) != 0xFFFFFFFFFFFFFFFF) {
            // Create difference mask by inverting equality mask
            uint64x2_t diff_mask = vreinterpretq_u64_u8(vmvnq_u8(equal_mask));
            uint64_t diff_low = vgetq_lane_u64(diff_mask, 0);
            uint64_t diff_high = vgetq_lane_u64(diff_mask, 1);

            // Find first differing byte by checking which 64-bit lane has differences
            if (diff_low != 0) {
                // Difference in lower 8 bytes: count trailing zeros and divide by 8
                int first_diff_byte_index = __builtin_ctzll(diff_low) / 8;
                return (unsigned char)s1[i + first_diff_byte_index] - (unsigned char)s2[i + first_diff_byte_index];
            } else {
                // Difference in upper 8 bytes: add 8 to account for lower half
                int first_diff_byte_index = 8 + (__builtin_ctzll(diff_high) / 8);
                return (unsigned char)s1[i + first_diff_byte_index] - (unsigned char)s2[i + first_diff_byte_index];
            }
        }
        i += 16;  // Move to next 16-byte chunk
    }
#endif

    // Handle remaining bytes or fallback for unsupported platforms
    return strncmp(s1 + i, s2 + i, n - i);
}

static inline const char* simd_strchr_any(const char* s, size_t n, const char* needles, size_t num_needles) {
    size_t i = 0;

// Only use NEON intrinsics on 64-bit ARM with compiler support
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    // Process 16-byte chunks using NEON 128-bit vector instructions
    while (i + 16 <= n) {
        // Load 16 bytes of haystack into vector
        uint8x16_t v_haystack = vld1q_u8((const uint8_t*)(s + i));

        // Initialize combined result mask to all zeros
        uint8x16_t combined_mask = vdupq_n_u8(0);

        // Check each needle character against all 16 haystack bytes
        for (size_t j = 0; j < num_needles; ++j) {
            // Broadcast single needle character to all 16 positions
            uint8x16_t v_needle = vdupq_n_u8(needles[j]);

            // Compare needle against all haystack bytes simultaneously
            uint8x16_t mask = vceqq_u8(v_haystack, v_needle);

            // OR the result with combined mask to accumulate matches
            combined_mask = vorrq_u8(combined_mask, mask);
        }

        // Convert to 64-bit view for easier bit manipulation
        uint64x2_t v_u64 = vreinterpretq_u64_u8(combined_mask);
        uint64_t low = vgetq_lane_u64(v_u64, 0);
        uint64_t high = vgetq_lane_u64(v_u64, 1);

        // Check if any match was found (any bit set in either lane)
        if ((low | high) != 0) {
            // Find position of first match by checking which 64-bit lane has matches
            if (low != 0) {
                // Match in lower 8 bytes: count trailing zeros and divide by 8
                int first_match_index = __builtin_ctzll(low) / 8;
                return s + i + first_match_index;
            } else {
                // Match in upper 8 bytes: add 8 to account for lower half
                int first_match_index = 8 + __builtin_ctzll(high) / 8;
                return s + i + first_match_index;
            }
        }
        i += 16;  // Move to next 16-byte chunk
    }
#endif

    // Handle remaining bytes (< 16) or fallback for unsupported platforms
    for (; i < n; ++i) {
        for (size_t j = 0; j < num_needles; ++j) {
            if (s[i] == needles[j]) return s + i;
        }
    }
    return NULL;
}


// -----------------------------
// Fallback Implementation (no SIMD support)
// -----------------------------
#else

// When no SIMD instruction sets are available, fall back to standard library functions
static inline int simd_strncmp(const char* s1, const char* s2, size_t n) {
    // Simply use the standard library implementation
    return strncmp(s1, s2, n);
}

static inline const char* simd_strchr_any(const char* s, size_t n, const char* needles, size_t num_needles) {
    // Use simple nested loops to check each character against each needle
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < num_needles; ++j) {
            if (s[i] == needles[j]) {
                return s + i;
            }
        }
    }
    return NULL;
}

#endif

#endif // SIMD_COMPARE_H
