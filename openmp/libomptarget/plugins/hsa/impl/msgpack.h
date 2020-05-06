#ifndef MSGPACK_H
#define MSGPACK_H

#include <functional>

namespace msgpack {

// The message pack format is dynamically typed, schema-less. Format is:
// message: [type][header][payload]
// where type is one byte, header length is a fixed length function of type
// payload is zero to N bytes, with the length encoded in [type][header]

// Scalar fields include boolean, signed integer, float, string etc
// Composite types are sequences of messages
// Array field is [header][element][element]...
// Map field is [header][key][value][key][value]...

// Multibyte integer fields are big endian encoded
// The map key can be any message type
// Maps may contain duplicate keys
// Data is not uniquely encoded, e.g. integer "8" may be stored as one byte or
// in as many as nine, as signed or unsigned. Implementation defined.
// Similarly "foo" may embed the length in the type field or in multiple bytes

// This parser is structured as an iterator over a sequence of bytes.
// It calls a user provided function on each message in order to extract fields
// The default implementation for each scalar type is to do nothing. For map or
// arrays, the default implementation returns just after that message to support
// iterating to the next message, but otherwise has no effect.

struct byte_range {
  const unsigned char *start;
  const unsigned char *end;
};

namespace fallback {

const unsigned char *skip_next_message(const unsigned char *start,
                                       const unsigned char *end);

void nop_string(size_t, const unsigned char *);
void nop_signed(int64_t);
void nop_unsigned(uint64_t);
void nop_boolean(bool);
void nop_array_elements(byte_range);
void nop_map_elements(byte_range, byte_range);

const unsigned char *nop_map(uint64_t N, byte_range);
const unsigned char *nop_array(uint64_t N, byte_range);

const unsigned char *array(uint64_t N, byte_range,
                           std::function<void(byte_range)> callback);
const unsigned char *map(uint64_t N, byte_range,
                         std::function<void(byte_range, byte_range)> callback);
} // namespace fallback

struct functors {

  std::function<void(size_t, const unsigned char *)> cb_string =
      fallback::nop_string;

  std::function<void(int64_t)> cb_signed = fallback::nop_signed;

  std::function<void(uint64_t)> cb_unsigned = fallback::nop_unsigned;

  std::function<void(bool)> cb_boolean = fallback::nop_boolean;

  std::function<void(byte_range, byte_range)> cb_map_elements =
      fallback::nop_map_elements;

  std::function<void(byte_range)> cb_array_elements =
      fallback::nop_array_elements;

  std::function<const unsigned char *(uint64_t N, byte_range)> cb_array =
      [=](uint64_t N, byte_range bytes) {
        return fallback::array(N, bytes, this->cb_array_elements);
      };

  std::function<const unsigned char *(uint64_t N, byte_range)>

      cb_map = [=](uint64_t N, byte_range bytes) {
        return fallback::map(N, bytes, this->cb_map_elements);
      };
};

typedef enum : uint8_t {
#define X(NAME, WIDTH, PAYLOAD, LOWER, UPPER) NAME,
#include "msgpack.def"
#undef X
} type;

[[noreturn]] void internal_error();
type parse_type(unsigned char x);
unsigned bytes_used_fixed(type ty);

typedef uint64_t (*payload_info_t)(const unsigned char *);
payload_info_t payload_info(msgpack::type ty);

template <typename T, typename R> R bitcast(T x);

template <msgpack::type ty>
const unsigned char *handle_msgpack_given_type(byte_range bytes, functors f) {
  const unsigned char *start = bytes.start;
  const unsigned char *end = bytes.end;
  const uint64_t available = end - start;
  assert(available != 0);

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
    return f.cb_array(N, {start + bytes_used, end});
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

inline const unsigned char *handle_msgpack(byte_range bytes, functors f) {
  const unsigned char *start = bytes.start;
  const unsigned char *end = bytes.end;
  const uint64_t available = end - start;
  if (available == 0) {
    return 0;
  }
  const type ty = parse_type(*start);

  switch (ty) {
#define X(NAME, WIDTH, PAYLOAD, LOWER, UPPER)                                  \
  case msgpack::NAME:                                                          \
    return handle_msgpack_given_type<msgpack::NAME>(bytes, f);
#include "msgpack.def"
#undef X
  }

  internal_error();
}

bool message_is_string(byte_range bytes, const char *str);

void foreach_map(byte_range,
                 std::function<void(byte_range, byte_range)> callback);

void foreach_array(byte_range, std::function<void(byte_range)> callback);

// Crude approximation to json
void dump(byte_range);

} // namespace msgpack

#endif
