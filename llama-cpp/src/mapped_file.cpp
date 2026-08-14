#include "mapped_file.h"

#include <fcntl.h>
#include <stdexcept>
#ifdef _WIN32
#include <fstream>
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#endif

#ifdef _WIN32
MappedFile::MappedFile(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::in | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to read the file: " + file_path + ".");
    }
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    try {
        buffer.resize((size_t)file_size);
    }
    catch (const std::bad_alloc& e) {
        throw std::runtime_error(
            "Insufficient memory to load model weights (size: " + std::to_string(file_size) + " bytes)."
        );
    }
    file.read(
        reinterpret_cast<uint8_t*>(buffer.data()), file_size * sizeof(uint8_t)
    );
    mapped_data = buffer.data();
    mapped_size = (size_t)file_size;
}

MappedFile::~MappedFile() {
    buffer.clear();
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : mapped_data(other.mapped_data), mapped_size(other.mapped_size) {
    buffer = std::move(other.buffer);
    other.mapped_data = nullptr;
    other.mapped_size = 0;
    other.buffer.clear();
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    mapped_data = other.mapped_data;
    mapped_size = other.mapped_size;
    buffer = std::move(other.buffer);
    other.mapped_data = nullptr;
    other.mapped_size = 0;
    other.buffer.clear();
    return *this;
}

#else
MappedFile::MappedFile(const std::string& file_path) {

    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Failed to read the file: " + file_path + ".");
    }
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Fstat failed");
    }
    size_t file_size = sb.st_size;
    void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        throw std::runtime_error("Mmap failed");
    }
    mapped_data = data;
    mapped_size = file_size;
}

MappedFile::~MappedFile() {
    if (mapped_data != nullptr && mapped_data != MAP_FAILED) {
        munmap(mapped_data, mapped_size);
    }
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : mapped_data(other.mapped_data), mapped_size(other.mapped_size) {
    other.mapped_data = nullptr;
    other.mapped_size = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (mapped_data != nullptr && mapped_data != MAP_FAILED) {
        munmap(mapped_data, mapped_size);
    }
    mapped_data = other.mapped_data;
    mapped_size = other.mapped_size;
    other.mapped_data = nullptr;
    other.mapped_size = 0;
    return *this;
}
#endif

const uint8_t* MappedFile::data() const {
    return reinterpret_cast<const uint8_t*>(mapped_data);
}

size_t MappedFile::size() const {
    return mapped_size;
}