#include "util/utils.cuh"
#include <gtest/gtest.h>
TEST(Util, bytes_equal) {
  ASSERT_FALSE(util::bytes_equal(reinterpret_cast<const uint8_t *>("12345"), 5,
                                 reinterpret_cast<const uint8_t *>("12"), 2));
  ASSERT_FALSE(util::bytes_equal(reinterpret_cast<const uint8_t *>("123"), 3,
                                 nullptr, 0));
  ASSERT_FALSE(util::bytes_equal(reinterpret_cast<const uint8_t *>("12335"), 5,
                                 reinterpret_cast<const uint8_t *>("12345"),
                                 5));
  ASSERT_TRUE(util::bytes_equal(reinterpret_cast<const uint8_t *>(""), 0, 
                                nullptr, 0));
  ASSERT_TRUE(util::bytes_equal(reinterpret_cast<const uint8_t *>("12345"), 5,
                                reinterpret_cast<const uint8_t *>("12345"), 5));
}