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

const unsigned char *handle_msgpack(byte_range, functors f);

bool message_is_string(byte_range bytes, const char *str);

void foreach_map(byte_range,
                 std::function<void(byte_range, byte_range)> callback);

void foreach_array(byte_range, std::function<void(byte_range)> callback);

// Crude approximation to json
void dump(byte_range);

} // namespace msgpack

#endif
