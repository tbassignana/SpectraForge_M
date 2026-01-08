/**
 * SpectraForge Metal - Image I/O
 *
 * Simple PNG and HDR file output without external dependencies.
 * Uses STB-style single-header approach for portability.
 */

#include "../include/spectraforge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// MINIMAL PNG ENCODER
// ============================================================================

// CRC32 lookup table
static unsigned int crc_table[256];
static int crc_table_computed = 0;

static void make_crc_table(void) {
    for (int n = 0; n < 256; n++) {
        unsigned int c = (unsigned int)n;
        for (int k = 0; k < 8; k++) {
            if (c & 1)
                c = 0xedb88320U ^ (c >> 1);
            else
                c = c >> 1;
        }
        crc_table[n] = c;
    }
    crc_table_computed = 1;
}

static unsigned int update_crc(unsigned int crc, const unsigned char* buf, size_t len) {
    if (!crc_table_computed) make_crc_table();

    unsigned int c = crc;
    for (size_t n = 0; n < len; n++) {
        c = crc_table[(c ^ buf[n]) & 0xff] ^ (c >> 8);
    }
    return c;
}

static unsigned int crc32(const unsigned char* buf, size_t len) {
    return update_crc(0xffffffffU, buf, len) ^ 0xffffffffU;
}

// Adler-32 checksum for DEFLATE
static unsigned int adler32(const unsigned char* data, size_t len) {
    unsigned int a = 1, b = 0;
    for (size_t i = 0; i < len; i++) {
        a = (a + data[i]) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

// Write big-endian 32-bit integer
static void write_be32(unsigned char* p, unsigned int v) {
    p[0] = (v >> 24) & 0xff;
    p[1] = (v >> 16) & 0xff;
    p[2] = (v >> 8) & 0xff;
    p[3] = v & 0xff;
}

// Write a PNG chunk
static int write_chunk(FILE* f, const char* type, const unsigned char* data, size_t len) {
    unsigned char header[8];
    write_be32(header, (unsigned int)len);
    memcpy(header + 4, type, 4);

    if (fwrite(header, 1, 8, f) != 8) return -1;
    if (len > 0 && fwrite(data, 1, len, f) != len) return -1;

    // CRC covers type + data
    unsigned int crc = crc32((const unsigned char*)type, 4);
    if (len > 0) {
        crc = update_crc(crc ^ 0xffffffffU, data, len) ^ 0xffffffffU;
    }

    unsigned char crc_buf[4];
    write_be32(crc_buf, crc);
    if (fwrite(crc_buf, 1, 4, f) != 4) return -1;

    return 0;
}

/**
 * Save RGB float data as PNG file.
 *
 * Uses uncompressed DEFLATE (store) for simplicity.
 * This creates larger files but avoids zlib dependency.
 */
int sf_save_png(const char* filename, const float* rgb_data, uint32_t width, uint32_t height) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return -1;
    }

    // PNG signature
    const unsigned char signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (fwrite(signature, 1, 8, f) != 8) {
        fclose(f);
        return -1;
    }

    // IHDR chunk
    unsigned char ihdr[13];
    write_be32(ihdr + 0, width);
    write_be32(ihdr + 4, height);
    ihdr[8] = 8;   // bit depth
    ihdr[9] = 2;   // color type (RGB)
    ihdr[10] = 0;  // compression
    ihdr[11] = 0;  // filter
    ihdr[12] = 0;  // interlace

    if (write_chunk(f, "IHDR", ihdr, 13) != 0) {
        fclose(f);
        return -1;
    }

    // Prepare raw image data with filter bytes
    size_t row_bytes = width * 3 + 1;  // +1 for filter byte
    size_t raw_size = row_bytes * height;
    unsigned char* raw_data = (unsigned char*)malloc(raw_size);

    if (!raw_data) {
        fclose(f);
        return -1;
    }

    // Convert float RGB to 8-bit with gamma correction
    for (uint32_t y = 0; y < height; y++) {
        unsigned char* row = raw_data + y * row_bytes;
        row[0] = 0;  // No filter

        for (uint32_t x = 0; x < width; x++) {
            size_t src_idx = (y * width + x) * 3;

            // Apply gamma correction (assuming linear input)
            float r = powf(fmaxf(0.0f, rgb_data[src_idx + 0]), 1.0f / 2.2f);
            float g = powf(fmaxf(0.0f, rgb_data[src_idx + 1]), 1.0f / 2.2f);
            float b = powf(fmaxf(0.0f, rgb_data[src_idx + 2]), 1.0f / 2.2f);

            // Clamp and convert to 8-bit
            row[1 + x * 3 + 0] = (unsigned char)(fminf(r, 1.0f) * 255.0f + 0.5f);
            row[1 + x * 3 + 1] = (unsigned char)(fminf(g, 1.0f) * 255.0f + 0.5f);
            row[1 + x * 3 + 2] = (unsigned char)(fminf(b, 1.0f) * 255.0f + 0.5f);
        }
    }

    // Create IDAT chunk with uncompressed DEFLATE
    // Format: zlib header (2 bytes) + blocks + adler32 (4 bytes)

    // Calculate block structure for uncompressed DEFLATE
    // Each block: 1 byte header + 2 bytes len + 2 bytes nlen + data
    // Max block size is 65535 bytes

    size_t max_block = 65535;
    size_t num_blocks = (raw_size + max_block - 1) / max_block;
    size_t deflate_size = 2 + num_blocks * 5 + raw_size + 4;

    unsigned char* idat_data = (unsigned char*)malloc(deflate_size);
    if (!idat_data) {
        free(raw_data);
        fclose(f);
        return -1;
    }

    // zlib header (no compression)
    idat_data[0] = 0x78;  // CM=8 (deflate), CINFO=7 (32K window)
    idat_data[1] = 0x01;  // FCHECK (makes header divisible by 31)

    size_t pos = 2;
    size_t remaining = raw_size;
    size_t src_pos = 0;

    while (remaining > 0) {
        size_t block_size = remaining > max_block ? max_block : remaining;
        int is_final = (remaining <= max_block) ? 1 : 0;

        // Block header
        idat_data[pos++] = is_final;  // BFINAL, BTYPE=00 (stored)

        // LEN and NLEN (little-endian)
        idat_data[pos++] = block_size & 0xff;
        idat_data[pos++] = (block_size >> 8) & 0xff;
        idat_data[pos++] = ~block_size & 0xff;
        idat_data[pos++] = (~block_size >> 8) & 0xff;

        // Data
        memcpy(idat_data + pos, raw_data + src_pos, block_size);
        pos += block_size;
        src_pos += block_size;
        remaining -= block_size;
    }

    // Adler-32 checksum of uncompressed data
    unsigned int adler = adler32(raw_data, raw_size);
    write_be32(idat_data + pos, adler);
    pos += 4;

    if (write_chunk(f, "IDAT", idat_data, pos) != 0) {
        free(idat_data);
        free(raw_data);
        fclose(f);
        return -1;
    }

    free(idat_data);
    free(raw_data);

    // IEND chunk
    if (write_chunk(f, "IEND", NULL, 0) != 0) {
        fclose(f);
        return -1;
    }

    fclose(f);
    printf("Saved PNG: %s (%ux%u)\n", filename, width, height);
    return 0;
}

// ============================================================================
// HDR (RADIANCE) FORMAT
// ============================================================================

/**
 * Convert float RGB to RGBE format.
 */
static void float_to_rgbe(float r, float g, float b, unsigned char rgbe[4]) {
    float v = fmaxf(fmaxf(r, g), b);

    if (v < 1e-32f) {
        rgbe[0] = rgbe[1] = rgbe[2] = rgbe[3] = 0;
        return;
    }

    int e;
    float m = frexpf(v, &e) * 256.0f / v;

    rgbe[0] = (unsigned char)(r * m);
    rgbe[1] = (unsigned char)(g * m);
    rgbe[2] = (unsigned char)(b * m);
    rgbe[3] = (unsigned char)(e + 128);
}

/**
 * Save RGB float data as HDR (Radiance) file.
 */
int sf_save_hdr(const char* filename, const float* rgb_data, uint32_t width, uint32_t height) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return -1;
    }

    // Write header
    fprintf(f, "#?RADIANCE\n");
    fprintf(f, "FORMAT=32-bit_rle_rgbe\n");
    fprintf(f, "\n");
    fprintf(f, "-Y %u +X %u\n", height, width);

    // Write scanlines (uncompressed for simplicity)
    unsigned char rgbe[4];

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            size_t idx = (y * width + x) * 3;
            float_to_rgbe(rgb_data[idx], rgb_data[idx + 1], rgb_data[idx + 2], rgbe);
            fwrite(rgbe, 1, 4, f);
        }
    }

    fclose(f);
    printf("Saved HDR: %s (%ux%u)\n", filename, width, height);
    return 0;
}
