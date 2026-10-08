#ifndef METAL_TENSOR_H
#define METAL_TENSOR_H

#include <vector>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

// Device buffers are shared, reference-counted allocations: MTensor copies alias the
// same memory and view() does not own it. With Metal they are MTLBuffers (retained
// through CoreFoundation); without Metal (the CPU backend) they are host blocks with
// the same semantics.
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
inline void mtensorRetain(void *b) { CFRetain(b); }
inline void mtensorRelease(void *b) { CFRelease(b); }
#else
struct MTensorHostBuffer {
    std::atomic<long> refs{1};
    void *data = nullptr;
};
inline void mtensorRetain(void *b) {
    static_cast<MTensorHostBuffer*>(b)->refs.fetch_add(1, std::memory_order_relaxed);
}
inline void mtensorRelease(void *b) {
    auto *h = static_cast<MTensorHostBuffer*>(b);
    if (h->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::free(h->data);
        delete h;
    }
}
#endif

// Forward-declare the Metal buffer type for C++ compatibility.
// Full Metal/Metal.h is only needed in .mm files.
#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* MTLBufferRef;  // opaque handle in pure C++
#endif

enum class DType : uint8_t {
    Float32,
    Int32,
    Int64,
    UInt8,
    Float64,
};

inline size_t dtypeSize(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        case DType::UInt8:   return 1;
        case DType::Float64: return 8;
    }
    return 0;
}

// Lightweight GPU tensor — wraps an MTLBuffer with shape metadata.
class MTensor {
public:
    MTensor() = default;

#ifdef __OBJC__
    // GPU allocation (Objective-C++ only)
    MTensor(id<MTLDevice> device, std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        if (bytes == 0) bytes = 4;
        id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) {
            throw std::runtime_error(
                "msplat: MTLBuffer allocation failed for " + std::to_string(bytes) +
                " bytes; device already holds " +
                std::to_string((size_t)[device currentAllocatedSize]));
        }
        _buffer = (__bridge_retained void*)buf;
        _ownsBuffer = true;
        _data = [buf contents];  // cache CPU-accessible pointer for C++ access
    }

    id<MTLBuffer> buffer() const { return (__bridge id<MTLBuffer>)_buffer; }
#endif

#ifndef __APPLE__
    // Device allocation without Metal: a shared, zero-filled host block (gpu_empty /
    // gpu_zeros in the CPU backend)
    struct HostDevice {};
    MTensor(HostDevice, std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        if (bytes == 0) bytes = 4;
        auto *h = new MTensorHostBuffer;
        h->data = std::calloc(1, bytes);
        if (!h->data) {
            delete h;
            throw std::runtime_error("msplat: host allocation failed for " + std::to_string(bytes) + " bytes");
        }
        _buffer = h;
        _ownsBuffer = true;
        _data = h->data;
    }
#endif

    // CPU allocation (no Metal buffer)
    MTensor(std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        _cpu_data.resize(bytes);
    }

    ~MTensor() { releaseBuffer(); }

    MTensor(const MTensor &o)
        : _buffer(o._buffer), _data(o._data), _cpu_data(o._cpu_data),
          _shape(o._shape), _dtype(o._dtype), _numel(o._numel),
          _ownsBuffer(o._ownsBuffer) {
        if (_buffer && _ownsBuffer) mtensorRetain(_buffer);
    }

    MTensor(MTensor &&o) noexcept
        : _buffer(o._buffer), _data(o._data), _cpu_data(std::move(o._cpu_data)),
          _shape(std::move(o._shape)), _dtype(o._dtype), _numel(o._numel),
          _ownsBuffer(o._ownsBuffer) {
        o._buffer = nullptr; o._data = nullptr; o._numel = 0; o._ownsBuffer = false;
    }

    MTensor& operator=(const MTensor &o) {
        if (this == &o) return *this;
        void *incoming = o._buffer;
        bool incomingOwns = o._ownsBuffer;
        if (incoming && incomingOwns) mtensorRetain(incoming);  // retain before release: o may alias us
        releaseBuffer();
        _buffer = incoming; _ownsBuffer = incomingOwns;
        _data = o._data; _cpu_data = o._cpu_data;
        _shape = o._shape; _dtype = o._dtype; _numel = o._numel;
        return *this;
    }

    MTensor& operator=(MTensor &&o) noexcept {
        if (this == &o) return *this;
        releaseBuffer();
        _buffer = o._buffer; _ownsBuffer = o._ownsBuffer;
        _data = o._data; _cpu_data = std::move(o._cpu_data);
        _shape = std::move(o._shape); _dtype = o._dtype; _numel = o._numel;
        o._buffer = nullptr; o._data = nullptr; o._numel = 0; o._ownsBuffer = false;
        return *this;
    }

    bool defined() const { return _buffer != nullptr || !_cpu_data.empty(); }
    bool isGpu() const { return _buffer != nullptr; }

    int64_t numel() const { return _numel; }
    int64_t size(int dim) const {
        if (dim < 0) dim += _shape.size();
        return _shape[dim];
    }
    int ndim() const { return (int)_shape.size(); }
    const std::vector<int64_t>& shape() const { return _shape; }
    DType dtype() const { return _dtype; }
    size_t elementSize() const { return dtypeSize(_dtype); }
    size_t nbytes() const { return _numel * dtypeSize(_dtype); }

    void* data_ptr() {
        if (_data) return _data;
        return _cpu_data.data();
    }
    const void* data_ptr() const {
        if (_data) return _data;
        return _cpu_data.data();
    }

    template<typename T> T* data() { return static_cast<T*>(data_ptr()); }
    template<typename T> const T* data() const { return static_cast<const T*>(data_ptr()); }

    void zero() {
        memset(data_ptr(), 0, _numel * dtypeSize(_dtype));
    }

    // Create a CPU copy of the data
    MTensor cpu() const {
        MTensor out(_shape, _dtype);
        memcpy(out.data_ptr(), data_ptr(), nbytes());
        return out;
    }

    void reset() {
        releaseBuffer();
        _data = nullptr;
        _cpu_data.clear();
        _shape.clear();
        _numel = 0;
    }

    // Stride for dim 0 (elements per row)
    int64_t stride0() const {
        if (_shape.size() <= 1) return 1;
        int64_t s = 1;
        for (size_t i = 1; i < _shape.size(); i++) s *= _shape[i];
        return s;
    }

    // Create a view of the first `n` elements along dim 0.
    // WARNING: Non-owning — shares the underlying MTLBuffer without retaining it.
    // The caller MUST ensure the parent MTensor outlives all views.
    // Use-after-free if the parent is destroyed while a view exists.
    MTensor view(int64_t n) const {
        MTensor v;
        v._buffer = _buffer;  // shares the buffer (non-owning)
        v._data = _data;      // shares the CPU-accessible pointer
        v._shape = _shape;
        v._shape[0] = n;
        v._dtype = _dtype;
        v._numel = n * stride0();
        v._ownsBuffer = false;  // stays non-owning through copies
        return v;
    }

private:
    void releaseBuffer() {
        if (_buffer && _ownsBuffer) mtensorRelease(_buffer);
        _buffer = nullptr;
        _ownsBuffer = false;
    }

    void* _buffer = nullptr;  // id<MTLBuffer> as void*, retained when _ownsBuffer
    void* _data = nullptr;    // cached CPU-accessible pointer (shared memory on Apple Silicon)
    std::vector<uint8_t> _cpu_data;
    std::vector<int64_t> _shape;
    DType _dtype = DType::Float32;
    int64_t _numel = 0;
    bool _ownsBuffer = false;
};

// Factory helpers (Objective-C++ only)
#ifdef __OBJC__
inline MTensor mtensor_empty(id<MTLDevice> dev, std::vector<int64_t> shape, DType dt) {
    return MTensor(dev, std::move(shape), dt);
}

inline MTensor mtensor_zeros(id<MTLDevice> dev, std::vector<int64_t> shape, DType dt) {
    MTensor t(dev, std::move(shape), dt);
    t.zero();
    return t;
}
#endif

#endif
