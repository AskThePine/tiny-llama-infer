#pragma once
#include "model.h"
#include "tokenizer.h"

constexpr uint32_t MAGIC_NUM = 0x70696E65; // pine
constexpr int32_t VERSION = 2;
constexpr size_t HEADER_SIZE = 256;

constexpr uint8_t MODEL = 0;
constexpr uint8_t TOKENIZER = 1;

void load_transformer(const std::string& file_path, Transformer& transformer);

void load_tokenizer(const std::string& file_path, Tokenizer& tokenizer);