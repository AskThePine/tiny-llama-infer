#include "loader.h"
#include "sampler.h"
#include "generator.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdlib>

int main(int argc, char* argv[]) {
    // default parameters
    std::string checkpoint_path;
    std::string tokenizer_path;
    std::string prompt;
    float temperature = 1.0f;
    float top_p = 0.9f;
    int32_t max_gen_len = 64;
    bool echo = false;
    uint32_t seed = 42;

    try {
        // parse parameters
        for (int i = 1;i < argc;i++) {
            std::string arg = argv[i];
            if (arg == "-checkpoint" || arg == "-ckpt") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("-ckpt is missing parameters.");
                }
                checkpoint_path = argv[++i];
            } else if (arg == "-tokenizer" || arg == "-t") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("-tokenizer is missing parameters.");
                }
                tokenizer_path = argv[++i];
            } else if (arg == "-prompt" || arg == "-p") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("-prompt is missing parameters.");
                }
                prompt = argv[++i];
            } else if (arg == "-temperature" || arg == "-T") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("-temperature is missing parameters.");
                }
                temperature = std::stof(std::string(argv[++i]));
            } else if (arg == "-topp" || arg == "-P") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("-topp is missing parameters.");
                }
                top_p = std::stof(std::string(argv[++i]));
            } else if (arg == "-len" || arg == "-l") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("-len is missing parameters.");
                }
                max_gen_len = std::stoi(std::string(argv[++i]));
            } else if (arg == "-echo" || arg == "-e") {
                echo = true;
            } else if (arg == "-seed" || arg == "-s") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("-seed is missing parameters.");
                }
                seed = static_cast<uint32_t>(std::stoul(std::string(argv[++i])));
            } else {
                throw std::runtime_error("Unknown option: " + arg + ".");
            }
        }

        // check parameters
        if (checkpoint_path.empty()) {
            throw std::runtime_error("Transformer path not found. Use -ckpt to set.");
        }
        if (tokenizer_path.empty()) {
            throw std::runtime_error("Tokenizer path not found. Use -t to set.");
        }
        if (prompt.empty()) {
            throw std::runtime_error("Prompt not found. Use -p to set.");
        }

        // load tokenizer and transformer from binary files
        Tokenizer tokenizer{};
        load_tokenizer(tokenizer_path, tokenizer);
        Transformer model{};
        load_transformer(checkpoint_path, model);

        // check vocab size and context length
        if (model.config.vocab_size != tokenizer.config.vocab_size) {
            throw std::runtime_error(
                "Model vocab size: " + std::to_string(model.config.vocab_size) +
                " is not equal to tokenizer vocab size: " + std::to_string(tokenizer.config.vocab_size) + "."
            );
        }
        if (model.config.max_seq_len != tokenizer.config.model_max_length) {
            throw std::runtime_error(
                "Model context lenght: " + std::to_string(model.config.max_seq_len) +
                " is not equal to tokenizer context lenght: " + std::to_string(tokenizer.config.model_max_length) + "."
            );
        }
        // build sampler and generator
        Sampler sampler(model.config.vocab_size, temperature, top_p, seed);
        Generator generator(model, tokenizer);
        // generate the text
        std::string out = generator.complete(prompt, sampler, max_gen_len, echo);
        std::cout << out << "\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
