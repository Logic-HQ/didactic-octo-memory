// Add this inside the public section of include/engine.hpp:

struct EngineMetadata {
    size_t total_layers;
    size_t scratchpad_capacity_bytes;
};

EngineMetadata get_metadata() const {
    return { layers_.size(), allocator_.get_buffer_capacity() };
}
