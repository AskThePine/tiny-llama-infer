#pragma once
#include <string>
#include <vector>

struct MappedFile {
    void* mapped_data = nullptr;
    size_t mapped_size = 0;
#ifdef _WIN32
    std::vector<uint8_t> buffer;
#endif

    MappedFile() = default;
    MappedFile(const std::string& file_path);
    ~MappedFile();

    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const uint8_t* data() const;
    size_t size() const;
};