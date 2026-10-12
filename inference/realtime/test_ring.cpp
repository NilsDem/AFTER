#include "BlockRing.hpp"
#include <cassert>
#include <iostream>
#include <thread>
int main() {
    BlockRing ring;
    float in[64], out[64];
    for (auto& x: in) x = .25f;
    assert(ring.push(in,64));
    auto* slot = ring.take(); assert(slot);
    std::memcpy(slot->output.data(),slot->input.data(),sizeof(in));
    assert(ring.finish(*slot)); assert(ring.pop(out,64));
    for (auto x: out) assert(x == .25f);
    assert(ring.push(in,64)); slot = ring.take(); assert(slot);
    assert(!ring.pop(out,64)); // inference missed its deadline
    for (auto x: out) assert(x == 0);
    assert(!ring.push(in,64)); // current input is discarded too
    assert(!ring.finish(*slot)); // late result can never play
    assert(ring.push(in,64)); slot = ring.take(); assert(slot);
    std::memcpy(slot->output.data(),in,sizeof(in));
    assert(ring.finish(*slot)); assert(ring.pop(out,64));
    assert(ring.dropped == 2);
    assert(ring.push(in,64)); // queued input expires without inference
    assert(!ring.pop(out,64)); assert(!ring.take());
    assert(!ring.push(in,64));
    for (int i=0; i<10000; ++i) {
        assert(ring.push(in,64)); slot = ring.take(); assert(slot);
        assert(!ring.pop(out,64)); assert(!ring.finish(*slot));
        assert(!ring.push(in,64));
    }
    BlockRing aligned;
    aligned.deadlineLead = 2;
    assert(aligned.push(in,64)); slot = aligned.take(); assert(slot);
    std::memcpy(slot->output.data(),in,sizeof(in));
    assert(aligned.finish(*slot));
    assert(!aligned.pop(out,64)); // intentional startup alignment, no drop
    assert(aligned.dropped == 0);
    assert(aligned.pop(out,64));
    for (auto x: out) assert(x == .25f);

    // Exercise all three independent owners under the thread sanitizer too.
    BlockRing concurrent;
    std::atomic<bool> done{false};
    std::thread producer([&] {
        float block[64];
        for (int i=1; i<=30000; ++i) {
            for (auto& x: block) x = float(i);
            concurrent.push(block,64);
            std::this_thread::yield();
        }
    });
    std::thread inference([&] {
        while (!done.load()) {
            if (auto* job = concurrent.take()) {
                for (auto x: job->input) { (void)x; } // worker owns the payload
                std::memcpy(job->output.data(),job->input.data(),sizeof(in));
                concurrent.finish(*job);
            } else std::this_thread::yield();
        }
    });
    for (int i=0; i<30000; ++i) {
        concurrent.pop(out,64);
        for (auto x: out) assert(x == out[0]);
        std::this_thread::yield();
    }
    producer.join(); done.store(true); inference.join();
    std::cout << "Ring deadline, silence, input discard and recovery tests passed\n";
}
