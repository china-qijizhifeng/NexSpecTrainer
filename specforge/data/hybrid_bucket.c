/* Stable counting partition for the codec's <=512 integer contexts.
 * No tensor arithmetic: preserve the exact order within every context. */
#include <stdint.h>
#include <stddef.h>

int hybrid_bucket(const uint8_t *values, const uint16_t *contexts,
                  size_t length, size_t buckets, uint8_t *output,
                  int64_t *counts) {
    if (buckets == 0 || buckets > 512) return -1;
    uint64_t offsets[512];
    for (size_t c = 0; c < buckets; ++c) counts[c] = 0;
    for (size_t i = 0; i < length; ++i) {
        if (contexts[i] >= buckets) return -2;
        ++counts[contexts[i]];
    }
    uint64_t end = 0;
    for (size_t c = 0; c < buckets; ++c) {
        offsets[c] = end;
        end += counts[c];
    }
    for (size_t i = 0; i < length; ++i)
        output[offsets[contexts[i]]++] = values[i];
    return 0;
}
