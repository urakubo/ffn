// Experimental Linux-only TensorFlow request-ID audit/interposition tool.
// Does not disable TensorFlow's duplicate-request detection or replay requests.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <unistd.h>

namespace {
std::atomic<uint64_t> calls{0};
std::mutex log_mutex;
using Generator = int64_t (*)();
std::atomic<Generator> registered_generator{nullptr};

[[noreturn]] void fail(const char* message) {
  std::fprintf(stderr, "FFN_ID_HOOK_FATAL: %s\n", message);
  std::_Exit(86);
}

Generator original() {
  if (Generator registered = registered_generator.load(std::memory_order_acquire))
    return registered;
  static Generator fn = [] {
    void* symbol = dlsym(RTLD_NEXT, "_ZN10tensorflow18GetUniqueRequestIdEv");
    if (!symbol) fail("TensorFlow GetUniqueRequestId symbol was not resolved");
    return reinterpret_cast<Generator>(symbol);
  }();
  return fn;
}

unsigned id_namespace() {
  static unsigned value = [] {
    const char* text = std::getenv("FFN_ID_NAMESPACE");
    if (!text || !*text) fail("FFN_ID_NAMESPACE must be set per process");
    char* end = nullptr;
    long n = std::strtol(text, &end, 10);
    if (*end || n < 1 || n > 32767) fail("namespace must be 1..32767");
    return static_cast<unsigned>(n);
  }();
  return value;
}

bool counter_mode() {
  static bool value = [] {
    const char* mode = std::getenv("FFN_ID_MODE");
    if (!mode || (std::strcmp(mode, "observe") && std::strcmp(mode, "counter")))
      fail("FFN_ID_MODE must be observe or counter");
    return !std::strcmp(mode, "counter");
  }();
  return value;
}

FILE* log_file() {
  static FILE* file = [] {
    const char* directory = std::getenv("FFN_ID_AUDIT_DIR");
    if (!directory || !*directory) fail("missing audit directory");
    char path[4096];
    int n = std::snprintf(path, sizeof(path), "%s/ids-%u-%ld.csv", directory,
                          id_namespace(), static_cast<long>(getpid()));
    if (n < 0 || n >= static_cast<int>(sizeof(path))) fail("audit path too long");
    FILE* result = std::fopen(path, "wx");
    if (!result) fail("cannot create exclusive audit log");
    // The supervisor may terminate a stalled process. Preserve every completed
    // log record without depending on process shutdown/destructor execution.
    std::setvbuf(result, nullptr, _IONBF, 0);
    std::fprintf(result, "sequence,original_id,returned_id\n");
    std::fprintf(stderr, "FFN_ID_HOOK_ACTIVE namespace=%u mode=%s pid=%ld\n",
                 id_namespace(), counter_mode() ? "counter" : "observe",
                 static_cast<long>(getpid()));
    return result;
  }();
  return file;
}
}

extern "C" void ffn_id_set_original(void* pointer) {
  if (!pointer) fail("null native generator registration");
  registered_generator.store(reinterpret_cast<Generator>(pointer),
                             std::memory_order_release);
}

extern "C" int64_t audited_request_id()
    asm("_ZN10tensorflow18GetUniqueRequestIdEv");
extern "C" int64_t audited_request_id() {
  const int64_t native_id = original()();
  const uint64_t sequence = calls.fetch_add(1, std::memory_order_relaxed) + 1;
  constexpr uint64_t limit = (uint64_t{1} << 48) - 1;
  if (sequence > limit) fail("request sequence exhausted");
  const int64_t returned = counter_mode()
      ? static_cast<int64_t>((uint64_t{id_namespace()} << 48) | sequence)
      : native_id;
  {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (std::fprintf(log_file(), "%llu,%lld,%lld\n",
                     static_cast<unsigned long long>(sequence),
                     static_cast<long long>(native_id),
                     static_cast<long long>(returned)) < 0)
      fail("audit log write failed");
  }
  return returned;
}

