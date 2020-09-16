#ifndef FIND_METADATA_HPP_INCLUDED
#define FIND_METADATA_HPP_INCLUDED

#include <cstddef>
#include <utility>

// Copied from libomptarget/plugins/hsa/impl/system.cpp
std::pair<unsigned char *, unsigned char *> find_metadata(void *binary,
                                                          size_t binSize);

#endif
