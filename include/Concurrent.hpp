#pragma once

#include "readerwriterqueue.h"

#include <atomic>

template<typename T>
struct ValueChannel {

    T read() {
        _fresh.store(false, std::memory_order_relaxed);
        return _value;
    }

    void write(const T &value) {
        _value = value;
        _fresh.store(true, std::memory_order_release);  // publishes _value
    }

    bool fresh() {
        return _fresh.load(std::memory_order_acquire);  // synchronizes-with write()'s release
    }

private:
    alignas(64) T _value;
    alignas(64) std::atomic<bool> _fresh = false;
};

template<class T> using QueueChannel = moodycamel::ReaderWriterQueue<T>;