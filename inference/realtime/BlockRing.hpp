#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>

// Ownership: input callback writes Free -> Writing -> Queued; worker owns
// Processing; output callback owns Reading. Expired processing remains owned by
// the worker until it acknowledges expiry. No allocation/locks in callbacks.
struct BlockRing {
    enum State { Free, Writing, Queued, Processing, Ready, Reading, Expired };
    struct Slot {
        std::atomic<int> state{Free};
        uint64_t deadline = 0, sequence = 0;
        std::array<float, 512> input{}, output{};
    };
    std::array<Slot, 3> slots;
    std::atomic<uint64_t> tick{0}, dropped{0}, silent{0}, submitted{0};
    std::atomic<bool> skipInput{false};
    uint64_t sequence = 0; // input callback only
    unsigned writeIndex = 0, readIndex = 0;
    unsigned deadlineLead = 1; // configured before starting callbacks

    bool push(const float* input, unsigned n) {
        ++sequence;
        if (skipInput.exchange(false)) { ++dropped; return false; }
        for (unsigned i = 0; i < slots.size(); ++i) {
            const auto index = (writeIndex + i) % slots.size();
            auto& slot = slots[index];
            int expected = Free;
            if (!slot.state.compare_exchange_strong(expected, Writing)) continue;
            slot.deadline = tick.load() + deadlineLead;
            slot.sequence = sequence;
            std::memcpy(slot.input.data(), input, n * sizeof(float));
            slot.state.store(Queued, std::memory_order_release);
            writeIndex = (index + 1) % slots.size();
            ++submitted;
            return true;
        }
        ++dropped;
        return false;
    }

    Slot* take() {
        for (unsigned i = 0; i < slots.size(); ++i) {
            const auto index = (readIndex + i) % slots.size();
            auto& slot = slots[index];
            int expected = Queued;
            if (slot.state.compare_exchange_strong(expected, Processing)) {
                readIndex = (index + 1) % slots.size();
                return &slot;
            }
        }
        return nullptr;
    }

    bool finish(Slot& slot) {
        int expected = Processing;
        // On failure acquire the callback's Expired transition before publishing
        // Free. This also orders its deadline read before the producer reuses it.
        if (slot.state.compare_exchange_strong(expected, Ready, std::memory_order_acq_rel)) return true;
        slot.state.store(Free, std::memory_order_release);
        return false;
    }

    bool pop(float* output, unsigned n) {
        std::memset(output, 0, n * sizeof(float));
        const auto now = tick.fetch_add(1) + 1;
        bool played = false, missed = false;
        for (auto& slot : slots) {
            int state = slot.state.load(std::memory_order_acquire);
            // A worker may move Queued -> Processing -> Ready while we inspect
            // a slot. Retry those two transitions so a just-ready block plays.
            while (state == Ready || state == Queued || state == Processing) {
                if (slot.deadline > now) break;
                if (state == Ready) {
                    if (!slot.state.compare_exchange_strong(state, Reading)) continue;
                    if (slot.deadline == now && !played) {
                        std::memcpy(output, slot.output.data(), n * sizeof(float));
                        played = true;
                    } else { ++dropped; missed = true; }
                    slot.state.store(Free, std::memory_order_release);
                } else {
                    if (!slot.state.compare_exchange_strong(state, state == Queued ? Free : Expired)) continue;
                    ++dropped;
                    missed = true;
                }
                break;
            }
        }
        if (missed) skipInput.store(true);
        if (!played) ++silent;
        return played;
    }
};
static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(std::atomic<int>::is_always_lock_free);
