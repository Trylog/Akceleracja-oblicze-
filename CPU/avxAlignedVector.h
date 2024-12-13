#pragma once

#include <memory>
#include <vector>
#include <cstdlib> // For malloc and free
#include <new>     // For std::bad_alloc
#include <cstddef> // For std::size_t
#include <cassert> // For assert

// Custom aligned allocator
template <typename T>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U>&) {}

    T* allocate(std::size_t n) {
        // Calculate the total size to allocate
        std::size_t size = n * sizeof(T);
        // Allocate extra memory for alignment
        void* ptr = std::malloc(size + sizeof(std::uintptr_t) + 32);
        if (!ptr) {
            throw std::bad_alloc();
        }

        // Calculate aligned pointer
        std::uintptr_t aligned_ptr = reinterpret_cast<std::uintptr_t>(ptr) + sizeof(std::uintptr_t);
        aligned_ptr += (32 - (aligned_ptr % 32)) % 32; // Align to 32 bytes

        // Store the original pointer just before the aligned pointer for free
        *reinterpret_cast<std::uintptr_t*>(aligned_ptr - sizeof(std::uintptr_t)) = reinterpret_cast<std::uintptr_t>(ptr);

        return reinterpret_cast<T*>(aligned_ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        // Retrieve the original pointer and free it
        std::uintptr_t original_ptr = *reinterpret_cast<std::uintptr_t*>(reinterpret_cast<std::uintptr_t>(ptr) - sizeof(std::uintptr_t));
        std::free(reinterpret_cast<void*>(original_ptr));
    }
};

template <typename T>
using AvxAlignedMatrix = std::vector<std::vector<T, AlignedAllocator<T>>>;

template <typename T>
AvxAlignedMatrix<T> createAvxAlignedMatrix(int n, int m) {
    std::vector<std::vector<T, AlignedAllocator<T>>> matrix(n,
                                                            std::vector<T, AlignedAllocator<T>>(m)
    );

    return matrix;
}