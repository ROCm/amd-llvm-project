#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

#include "msgpack.h"

namespace {
[[noreturn]] void internal_error() {
  printf("internal error\n");
  exit(1);
}
} // namespace

namespace msgpack {

typedef enum : uint8_t {
#define X(NAME, WIDTH, PAYLOAD, LOWER, UPPER) NAME,
#include "msgpack.def"
#undef X
} type;

const char *type_name(type ty) {
  switch (ty) {
#define X(NAME, WIDTH, PAYLOAD, LOWER, UPPER)                                  \
  case NAME:                                                                   \
    return #NAME;
#include "msgpack.def"
#undef X
  }
  internal_error();
}

unsigned bytes_used_fixed(msgpack::type ty) {
  using namespace msgpack;
  switch (ty) {
#define X(NAME, WIDTH, PAYLOAD, LOWER, UPPER)                                  \
  case NAME:                                                                   \
    return WIDTH;
#include "msgpack.def"
#undef X
  }
  internal_error();
}

msgpack::type parse_type(unsigned char x) {

#define X(NAME, WIDTH, PAYLOAD, LOWER, UPPER)                                  \
  if (x >= LOWER && x <= UPPER) {                                              \
    return NAME;                                                               \
  } else
#include "msgpack.def"
#undef X
  { internal_error(); }
}

} // namespace msgpack

template <typename T, typename R> R bitcast(T x) {
  static_assert(sizeof(T) == sizeof(R), "");
  R tmp;
  memcpy(&tmp, &x, sizeof(T));
  return tmp;
}

// Helper functions for reading additional payload from the header
// Depending on the type, this can be a number of bytes, elements,
// key-value pairs or an embedded integer.
// Each takes a pointer to the start of the header and returns a uint64_t

typedef uint64_t (*payload_info_t)(const unsigned char *);

namespace {
namespace payload {
uint64_t read_zero(const unsigned char *) { return 0; }

// Read the first byte and zero/sign extend it
uint64_t read_embedded_u8(const unsigned char *start) { return start[0]; }
uint64_t read_embedded_s8(const unsigned char *start) {
  int64_t res = bitcast<uint8_t, int8_t>(start[0]);
  return bitcast<int64_t, uint64_t>(res);
}

// Read a masked part of the first byte
uint64_t read_via_mask_0x1(const unsigned char *start) { return *start & 0x1u; }
uint64_t read_via_mask_0xf(const unsigned char *start) { return *start & 0xfu; }
uint64_t read_via_mask_0x1f(const unsigned char *start) {
  return *start & 0x1fu;
}

// Read 1/2/4/8 bytes immediately following the type byte and zero/sign extend
// Big endian format.
uint64_t read_size_field_u8(const unsigned char *from) {
  from++;
  return from[0];
}

// TODO: detect whether host is little endian or not, and whether the intrinsic
// is available. And probably use the builtin to test the diy
const bool use_bswap = false;

uint64_t read_size_field_u16(const unsigned char *from) {
  from++;
  if (use_bswap) {
    uint16_t b;
    memcpy(&b, from, 2);
    return __builtin_bswap16(b);
  } else {
    return (from[0] << 8u) | from[1];
  }
}
uint64_t read_size_field_u32(const unsigned char *from) {
  from++;
  if (use_bswap) {
    uint32_t b;
    memcpy(&b, from, 4);
    return __builtin_bswap32(b);
  } else {
    return (from[0] << 24u) | (from[1] << 16u) | (from[2] << 8u) |
           (from[3] << 0u);
  }
}
uint64_t read_size_field_u64(const unsigned char *from) {
  from++;
  if (use_bswap) {
    uint64_t b;
    memcpy(&b, from, 8);
    return __builtin_bswap64(b);
  } else {
    return ((uint64_t)from[0] << 56u) | ((uint64_t)from[1] << 48u) |
           ((uint64_t)from[2] << 40u) | ((uint64_t)from[3] << 32u) |
           (from[4] << 24u) | (from[5] << 16u) | (from[6] << 8u) |
           (from[7] << 0u);
  }
}

uint64_t read_size_field_s8(const unsigned char *from) {
  uint8_t u = read_size_field_u8(from);
  int64_t res = bitcast<uint8_t, int8_t>(u);
  return bitcast<int64_t, uint64_t>(res);
}
uint64_t read_size_field_s16(const unsigned char *from) {
  uint16_t u = read_size_field_u16(from);
  int64_t res = bitcast<uint16_t, int16_t>(u);
  return bitcast<int64_t, uint64_t>(res);
}
uint64_t read_size_field_s32(const unsigned char *from) {
  uint32_t u = read_size_field_u32(from);
  int64_t res = bitcast<uint32_t, int32_t>(u);
  return bitcast<int64_t, uint64_t>(res);
}
uint64_t read_size_field_s64(const unsigned char *from) {
  uint64_t u = read_size_field_u64(from);
  int64_t res = bitcast<uint64_t, int64_t>(u);
  return bitcast<int64_t, uint64_t>(res);
}
} // namespace payload
} // namespace

payload_info_t payload_info(msgpack::type ty) {
  using namespace msgpack;
  switch (ty) {
#define X(NAME, WIDTH, PAYLOAD, LOWER, UPPER)                                  \
  case NAME:                                                                   \
    return payload::PAYLOAD;
#include "msgpack.def"
#undef X
  }
  internal_error();
}

// Only failure mode is going to be out of bounds
// Return NULL on out of bounds, otherwise start of the next entry

namespace fallback {

void nop_string(size_t, const unsigned char *) {}
void nop_signed(int64_t) {}
void nop_unsigned(uint64_t) {}
void nop_boolean(bool) {}
void nop_array_elements(byte_range) {}
void nop_map_elements(byte_range, byte_range) {}

const unsigned char *array(uint64_t N, byte_range bytes,
                           std::function<void(byte_range)> callback) {
  for (uint64_t i = 0; i < N; i++) {
    const unsigned char *next = skip_next_message(bytes.start, bytes.end);
    if (!next) {
      return 0;
    }
    callback(bytes);

    bytes.start = next;
  }
  return bytes.start;
}

const unsigned char *map(uint64_t N, byte_range bytes,
                         std::function<void(byte_range, byte_range)> callback) {

  for (uint64_t i = 0; i < N; i++) {
    const unsigned char *start_key = bytes.start;
    const unsigned char *end_key = skip_next_message(start_key, bytes.end);

    if (!end_key) {
      break;
    }

    const unsigned char *start_value = end_key;
    const unsigned char *end_value = skip_next_message(start_value, bytes.end);

    if (!end_value) {
      break;
    }

    callback({start_key, end_key}, {start_value, end_value});

    bytes.start = end_value;
  }
  return bytes.start;
}

const unsigned char *nop_map(uint64_t N, byte_range bytes) {
  return map(N, bytes, nop_map_elements);
}

const unsigned char *nop_array(uint64_t N, byte_range bytes) {
  return array(N, bytes, nop_array_elements);
}

} // namespace fallback

const unsigned char *fallback::skip_next_message(const unsigned char *start,
                                                 const unsigned char *end) {
  return handle_msgpack({start, end}, {});
}

const unsigned char *handle_msgpack(byte_range bytes, functors f) {
  const unsigned char *start = bytes.start;
  const unsigned char *end = bytes.end;
  const uint64_t available = end - start;
  if (available == 0) {
    return 0;
  }
  const msgpack::type ty = msgpack::parse_type(*start);
  const uint64_t bytes_used = bytes_used_fixed(ty);
  if (available < bytes_used) {
    return 0;
  }
  const uint64_t available_post_header = available - bytes_used;

  const payload_info_t info = payload_info(ty);
  const uint64_t N = info(start);

  switch (ty) {
  case msgpack::t:
  case msgpack::f: {
    // t is 0b11000010, f is 0b11000011, masked with 0x1
    f.cb_boolean(N);
    return start + bytes_used;
  }

  case msgpack::posfixint:
  case msgpack::uint8:
  case msgpack::uint16:
  case msgpack::uint32:
  case msgpack::uint64: {
    f.cb_unsigned(N);
    return start + bytes_used;
  }

  case msgpack::negfixint:
  case msgpack::int8:
  case msgpack::int16:
  case msgpack::int32:
  case msgpack::int64: {
    f.cb_signed(bitcast<uint64_t, int64_t>(N));
    return start + bytes_used;
  }

  case msgpack::fixstr:
  case msgpack::str8:
  case msgpack::str16:
  case msgpack::str32: {
    if (available_post_header < N) {
      return 0;
    } else {
      f.cb_string(N, start + bytes_used);
      return start + bytes_used + N;
    }
  }

  case msgpack::fixarray:
  case msgpack::array16:
  case msgpack::array32: {
    return f.cb_array(N, byte_range{start + bytes_used, end});
  }

  case msgpack::fixmap:
  case msgpack::map16:
  case msgpack::map32: {
    return f.cb_map(N, {start + bytes_used, end});
  }

  case msgpack::nil:
  case msgpack::bin8:
  case msgpack::bin16:
  case msgpack::bin32:
  case msgpack::float32:
  case msgpack::float64:
  case msgpack::ext8:
  case msgpack::ext16:
  case msgpack::ext32:
  case msgpack::fixext1:
  case msgpack::fixext2:
  case msgpack::fixext4:
  case msgpack::fixext8:
  case msgpack::fixext16:
  case msgpack::never_used: {
    if (available_post_header < N) {
      return 0;
    }
    return start + bytes_used + N;
  }
  }
  internal_error();
}

bool message_is_string(byte_range bytes, const char *needle) {
  bool matched = false;
  functors f;
  size_t needleN = strlen(needle);

  f.cb_string = [=, &matched](size_t N, const unsigned char *str) {
    if (N == needleN) {
      if (memcmp(needle, str, N) == 0) {
        matched = true;
      }
    }
  };

  handle_msgpack(bytes, f);
  return matched;
}

void foreach_map(byte_range bytes,
                 std::function<void(byte_range, byte_range)> callback) {
  functors f;
  f.cb_map_elements = callback;
  handle_msgpack(bytes, f);
}

void foreach_array(byte_range bytes, std::function<void(byte_range)> callback) {
  functors f;
  f.cb_array_elements = callback;
  handle_msgpack(bytes, f);
}

void msgpack::dump(byte_range bytes) {
  functors f;
  unsigned indent = 0;
  const unsigned by = 2;

  f.cb_string = [&](size_t N, const unsigned char *bytes) {
    char *tmp = (char *)malloc(N + 1);
    memcpy(tmp, bytes, N);
    tmp[N] = '\0';
    printf("\"%s\"", tmp);
    free(tmp);
  };

  f.cb_signed = [&](int64_t x) { printf("%ld", x); };
  f.cb_unsigned = [&](uint64_t x) { printf("%lu", x); };

  f.cb_array = [&](uint64_t N, byte_range bytes) -> const unsigned char * {
    printf("\n%*s[\n", indent, "");
    indent += by;

    for (uint64_t i = 0; i < N; i++) {
      indent += by;
      printf("%*s", indent, "");
      const unsigned char *next = handle_msgpack(bytes, f);
      printf(",\n");
      indent -= by;
      bytes.start = next;
      if (!next) {
        break;
      }
    }
    indent -= by;
    printf("%*s]", indent, "");

    return bytes.start;
  };

  f.cb_map = [&](uint64_t N, byte_range bytes) -> const unsigned char * {
    printf("\n%*s{\n", indent, "");
    indent += by;

    for (uint64_t i = 0; i < 2 * N; i += 2) {
      const unsigned char *start_key = bytes.start;
      printf("%*s", indent, "");
      const unsigned char *end_key = handle_msgpack({start_key, bytes.end}, f);
      if (!end_key) {
        break;
      }

      printf(" : ");

      const unsigned char *start_value = end_key;
      const unsigned char *end_value =
          handle_msgpack({start_value, bytes.end}, f);

      if (!end_value) {
        break;
      }

      printf(",\n");

      bytes.start = end_value;
    }

    indent -= by;
    printf("%*s}", indent, "");

    return bytes.start;
  };

  handle_msgpack(bytes, f);
  printf("\n");
}
