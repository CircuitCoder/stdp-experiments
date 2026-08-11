#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

constexpr uint32_t kInputs = 784;
constexpr uint32_t kExcitatory = 400;
constexpr uint32_t kWeightBytes = 4;
constexpr uint64_t kKiB = 1024ULL;
constexpr uint64_t kMiB = 1024ULL * 1024ULL;
constexpr uint64_t kEmptyTag = std::numeric_limits<uint64_t>::max();

enum class Kind { Sparse, Continuous, Barrier };
enum class Allocation { Eager, Deferred };
enum class Depression { WriteBack, Drain };
enum class Structure {
  Direct,
  SetLine,
  SetFine,
  HashLine,
  HashWeight,
  HashOffset
};

struct Event {
  Kind kind;
  uint32_t line = 0;
  uint32_t mask = 0;
};

struct Trace {
  uint32_t checkpoint = 0;
  uint64_t input_spikes = 0;
  uint64_t excitatory_spikes = 0;
  uint64_t spike_records = 0;
  uint64_t attempts = 0;
  std::vector<std::pair<std::vector<uint16_t>, std::vector<uint16_t>>> ticks;
  std::vector<size_t> barriers;
};

struct Metrics {
  uint64_t accesses = 0;
  uint64_t hits = 0;
  uint64_t logical_updates = 0;
  uint64_t sparse_updates = 0;
  uint64_t continuous_updates = 0;
  uint64_t reads = 0;
  uint64_t materialization_reads = 0;
  uint64_t writebacks = 0;
  uint64_t eviction_writebacks = 0;
  uint64_t drain_writebacks = 0;
  uint64_t barrier_writebacks = 0;
  uint64_t final_writebacks = 0;
  uint64_t operator_writebacks = 0;
  uint64_t written_words = 0;
  uint64_t forced_evictions = 0;
  uint64_t probes = 0;
  uint64_t max_probe = 0;
  uint64_t peak_resident_words = 0;
  uint64_t peak_deferred_words = 0;
};

struct Config {
  Structure structure = Structure::Direct;
  Allocation allocation = Allocation::Eager;
  Depression depression = Depression::WriteBack;
  uint32_t line_bytes = 64;
  uint64_t capacity_bytes = kMiB;
  uint32_t parameter = 1;
  bool flush_barriers = true;
};

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

uint32_t valid_mask(uint64_t line, uint32_t words_per_line) {
  const uint64_t start = line * words_per_line;
  const uint64_t total = uint64_t{kInputs} * kExcitatory;
  if (start >= total)
    return 0;
  const uint32_t count =
      static_cast<uint32_t>(std::min<uint64_t>(words_per_line, total - start));
  return count == 32 ? 0xffffffffU : ((uint32_t{1} << count) - 1U);
}

std::string name(Structure value) {
  switch (value) {
  case Structure::Direct:
    return "direct";
  case Structure::SetLine:
    return "set-line";
  case Structure::SetFine:
    return "set-fine";
  case Structure::HashLine:
    return "hash-line";
  case Structure::HashWeight:
    return "hash-weight";
  case Structure::HashOffset:
    return "hash-offset";
  }
  throw std::logic_error("invalid structure");
}

std::string name(Allocation value) {
  return value == Allocation::Eager ? "eager" : "deferred";
}
std::string name(Depression value) {
  return value == Depression::WriteBack ? "write-back" : "drain";
}

Structure parse_structure(std::string_view value) {
  if (value == "direct")
    return Structure::Direct;
  if (value == "set-line")
    return Structure::SetLine;
  if (value == "set-fine")
    return Structure::SetFine;
  if (value == "hash-line")
    return Structure::HashLine;
  if (value == "hash-weight")
    return Structure::HashWeight;
  if (value == "hash-offset")
    return Structure::HashOffset;
  throw std::runtime_error("unknown structure: " + std::string(value));
}

Allocation parse_allocation(std::string_view value) {
  if (value == "eager")
    return Allocation::Eager;
  if (value == "deferred")
    return Allocation::Deferred;
  throw std::runtime_error("allocation must be eager or deferred");
}

Depression parse_depression(std::string_view value) {
  if (value == "write-back")
    return Depression::WriteBack;
  if (value == "drain")
    return Depression::Drain;
  throw std::runtime_error("depression must be write-back or drain");
}

uint64_t parse_capacity(std::string value) {
  uint64_t multiplier = 1;
  if (value.ends_with("MiB")) {
    multiplier = kMiB;
    value.resize(value.size() - 3);
  } else if (value.ends_with("KiB")) {
    multiplier = 1024;
    value.resize(value.size() - 3);
  }
  return std::stoull(value) * multiplier;
}

Trace read_trace(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot open trace: " + path);
  Trace trace;
  std::string line;
  size_t tick_index = 0;
  while (std::getline(input, line)) {
    if (line.empty())
      continue;
    const auto record = json::parse(line);
    const std::string type = record.at("type");
    if (type == "metadata") {
      if (record.at("schema") != "stdp-firing-trace-v1" ||
          record.at("input_neurons") != kInputs ||
          record.at("excitatory_neurons") != kExcitatory ||
          record.at("input_delay_ms") != 0.0) {
        throw std::runtime_error("unsupported trace metadata");
      }
      trace.checkpoint = record.at("checkpoint");
    } else if (type == "attempt_start") {
      trace.barriers.push_back(tick_index);
      ++trace.attempts;
    } else if (type == "spikes") {
      std::vector<uint16_t> x;
      std::vector<uint16_t> e;
      for (const auto &id : record.at("X"))
        x.push_back(id.get<uint16_t>());
      for (const auto &id : record.at("E"))
        e.push_back(id.get<uint16_t>());
      trace.input_spikes += x.size();
      trace.excitatory_spikes += e.size();
      trace.ticks.emplace_back(std::move(x), std::move(e));
      ++trace.spike_records;
      ++tick_index;
    }
  }
  if (trace.checkpoint == 0)
    throw std::runtime_error("trace has no metadata record");
  return trace;
}

std::vector<Event> build_events(const Trace &trace, uint32_t line_bytes,
                                bool e_before_x = false) {
  const uint32_t words = line_bytes / kWeightBytes;
  std::vector<Event> result;
  const uint64_t estimate =
      trace.input_spikes * ((kExcitatory + words - 1) / words + 1) +
      trace.excitatory_spikes * kInputs + trace.attempts;
  result.reserve(estimate);
  size_t barrier_pos = 0;
  auto append_pre = [&](uint16_t pre) {
    const uint32_t begin = uint32_t{pre} * kExcitatory;
    const uint32_t end = begin + kExcitatory;
    uint32_t cursor = begin;
    while (cursor < end) {
      const uint32_t line = cursor / words;
      const uint32_t offset = cursor % words;
      const uint32_t count = std::min(words - offset, end - cursor);
      const uint32_t run =
          count == 32 ? 0xffffffffU : ((uint32_t{1} << count) - 1U);
      result.push_back(
          {Kind::Continuous, line, static_cast<uint32_t>(run << offset)});
      cursor += count;
    }
  };
  auto append_post = [&](uint16_t post) {
    for (uint32_t pre = 0; pre < kInputs; ++pre) {
      const uint32_t index = pre * kExcitatory + post;
      result.push_back(
          {Kind::Sparse, index / words, uint32_t{1} << (index % words)});
    }
  };
  for (size_t tick = 0; tick < trace.ticks.size(); ++tick) {
    while (barrier_pos < trace.barriers.size() &&
           trace.barriers[barrier_pos] == tick) {
      result.push_back({Kind::Barrier, 0, 0});
      ++barrier_pos;
    }
    const auto &[x, e] = trace.ticks[tick];
    if (e_before_x) {
      for (auto id : e)
        append_post(id);
      for (auto id : x)
        append_pre(id);
    } else {
      for (auto id : x)
        append_pre(id);
      for (auto id : e)
        append_post(id);
    }
  }
  while (barrier_pos++ < trace.barriers.size())
    result.push_back({Kind::Barrier, 0, 0});
  return result;
}

class Store {
public:
  Store(Config config, Metrics &metrics)
      : c_(config), m_(metrics), words_(config.line_bytes / kWeightBytes) {}
  virtual ~Store() = default;
  virtual void access(const Event &event) = 0;
  virtual void flush(bool final) = 0;
  virtual uint64_t metadata_bytes() const = 0;

protected:
  Config c_;
  Metrics &m_;
  uint32_t words_;

  void count_access(const Event &event) {
    ++m_.accesses;
    const uint64_t count = std::popcount(event.mask);
    m_.logical_updates += count;
    if (event.kind == Kind::Sparse)
      m_.sparse_updates += count;
    else
      m_.continuous_updates += count;
  }

  void write(uint32_t mask, bool has_operator, Kind cause) {
    ++m_.writebacks;
    m_.written_words += std::popcount(mask);
    if (has_operator)
      ++m_.operator_writebacks;
    if (cause == Kind::Continuous)
      ++m_.drain_writebacks;
    else if (cause == Kind::Barrier)
      ++m_.barrier_writebacks;
    else
      ++m_.eviction_writebacks;
  }
};

class DirectStore final : public Store {
public:
  using Store::Store;
  void access(const Event &event) override {
    count_access(event);
    const uint64_t updates = std::popcount(event.mask);
    m_.writebacks += updates;
    m_.written_words += updates;
    if (event.kind == Kind::Sparse) {
      if (c_.allocation == Allocation::Eager)
        m_.reads += updates;
      if (c_.allocation == Allocation::Deferred)
        m_.operator_writebacks += updates;
      m_.eviction_writebacks += updates;
    } else {
      m_.reads += updates;
      m_.drain_writebacks += updates;
    }
  }
  void flush(bool) override {}
  uint64_t metadata_bytes() const override { return 0; }
};

struct LineEntry {
  uint64_t tag = kEmptyTag;
  uint32_t resident = 0;
  uint32_t known = 0;
  uint32_t dirty = 0;
  uint64_t last = 0;
  uint64_t home = 0;
};

class LineStore final : public Store {
public:
  LineStore(Config config, Metrics &metrics)
      : Store(config, metrics),
        slots_(config.capacity_bytes / config.line_bytes), table_(slots_) {
    if (slots_ == 0)
      throw std::runtime_error("line store has zero slots");
    if (c_.structure == Structure::SetLine) {
      ways_ = c_.parameter;
      if (slots_ % ways_ != 0)
        throw std::runtime_error("capacity not divisible by ways");
      sets_ = slots_ / ways_;
    } else {
      probe_limit_ = c_.parameter;
    }
  }

  void access(const Event &event) override {
    count_access(event);
    ++clock_;
    auto pos = find(event.line);
    const bool resident_hit =
        pos.has_value() && (table_[*pos].resident & event.mask) == event.mask;
    if (resident_hit)
      ++m_.hits;

    if (event.kind == Kind::Continuous && c_.depression == Depression::Drain) {
      uint32_t resident = event.mask;
      uint32_t known = 0;
      uint32_t dirty = 0;
      if (pos) {
        resident |= table_[*pos].resident;
        known = table_[*pos].known;
        dirty = table_[*pos].dirty;
      }
      if ((known & event.mask) != event.mask) {
        ++m_.reads;
        ++m_.materialization_reads;
      }
      write(resident, (dirty & ~known) != 0, Kind::Continuous);
      if (pos)
        clear(*pos);
      update_peaks();
      return;
    }

    if (!pos) {
      LineEntry entry;
      entry.tag = event.line;
      entry.home = home(event.line);
      entry.last = clock_;
      if (event.kind == Kind::Sparse && c_.allocation == Allocation::Deferred) {
        entry.resident = event.mask;
      } else {
        ++m_.reads;
        if (event.kind == Kind::Continuous)
          ++m_.materialization_reads;
        entry.resident = valid_mask(event.line, words_);
        entry.known = entry.resident;
      }
      entry.dirty = event.mask;
      pos = insert(std::move(entry));
      if (!pos) {
        update_peaks();
        return;
      }
    } else {
      auto entry = table_[*pos];
      entry.last = clock_;
      if (event.kind == Kind::Continuous &&
          (entry.known & event.mask) != event.mask) {
        ++m_.reads;
        ++m_.materialization_reads;
        entry.resident = valid_mask(event.line, words_);
        entry.known = entry.resident;
      }
      entry.resident |= event.mask;
      entry.dirty |= event.mask;
      assign(*pos, entry);
    }
    update_peaks();
  }

  void flush(bool final) override {
    for (size_t i = 0; i < table_.size(); ++i) {
      auto &entry = table_[i];
      if (entry.tag == kEmptyTag)
        continue;
      if (entry.dirty) {
        ++m_.writebacks;
        m_.written_words += std::popcount(entry.resident);
        if (entry.dirty & ~entry.known)
          ++m_.operator_writebacks;
        if (final)
          ++m_.final_writebacks;
        else
          ++m_.barrier_writebacks;
      }
      entry = {};
      entry.tag = kEmptyTag;
    }
    resident_words_ = deferred_words_ = 0;
  }

  uint64_t metadata_bytes() const override { return slots_ * 29; }

private:
  uint64_t slots_;
  uint64_t ways_ = 1;
  uint64_t sets_ = 1;
  uint64_t probe_limit_ = 1;
  uint64_t clock_ = 0;
  uint64_t resident_words_ = 0;
  uint64_t deferred_words_ = 0;
  std::vector<LineEntry> table_;

  uint64_t home(uint64_t tag) const {
    if (c_.structure == Structure::SetLine)
      return splitmix64(tag) % sets_;
    return splitmix64(tag) % slots_;
  }

  std::optional<size_t> find(uint64_t tag) {
    if (c_.structure == Structure::SetLine) {
      const uint64_t begin = home(tag) * ways_;
      for (uint64_t way = 0; way < ways_; ++way) {
        ++m_.probes;
        if (table_[begin + way].tag == tag)
          return begin + way;
      }
      m_.max_probe = std::max(m_.max_probe, ways_);
    } else {
      const uint64_t begin = home(tag);
      for (uint64_t distance = 0; distance < probe_limit_; ++distance) {
        ++m_.probes;
        const size_t pos = (begin + distance) % slots_;
        if (table_[pos].tag == tag)
          return pos;
      }
      m_.max_probe = std::max(m_.max_probe, probe_limit_);
    }
    return std::nullopt;
  }

  std::optional<size_t> insert(LineEntry entry) {
    if (c_.structure == Structure::SetLine) {
      const size_t begin = home(entry.tag) * ways_;
      size_t victim = begin;
      for (size_t way = 0; way < ways_; ++way) {
        const size_t pos = begin + way;
        if (table_[pos].tag == kEmptyTag) {
          assign(pos, entry);
          return pos;
        }
        if (table_[pos].last < table_[victim].last)
          victim = pos;
      }
      evict(victim, false);
      assign(victim, entry);
      return victim;
    }

    const uint64_t target_tag = entry.tag;
    size_t pos = entry.home;
    uint64_t distance = 0;
    while (distance < probe_limit_) {
      ++m_.probes;
      if (table_[pos].tag == kEmptyTag) {
        assign(pos, entry);
        return find(target_tag);
      }
      const uint64_t occupant_distance =
          (pos + slots_ - table_[pos].home) % slots_;
      if (occupant_distance < distance) {
        LineEntry displaced = table_[pos];
        assign(pos, entry);
        entry = displaced;
        distance = occupant_distance;
      }
      pos = (pos + 1) % slots_;
      ++distance;
    }
    ++m_.forced_evictions;
    spill_displaced(entry);
    return find(target_tag);
  }

  void spill_displaced(const LineEntry &external) {
    uint32_t resident = external.resident;
    uint32_t known = external.known;
    uint32_t dirty = external.dirty;
    if (dirty)
      write(resident, (dirty & ~known) != 0, Kind::Sparse);
  }

  void evict(size_t pos, bool forced) {
    if (forced)
      ++m_.forced_evictions;
    const auto entry = table_[pos];
    if (entry.dirty)
      write(entry.resident, (entry.dirty & ~entry.known) != 0, Kind::Sparse);
    clear(pos);
  }

  void clear(size_t pos) { assign(pos, LineEntry{}); }

  void assign(size_t pos, const LineEntry &entry) {
    const auto &old = table_[pos];
    if (old.tag != kEmptyTag) {
      resident_words_ -= std::popcount(old.resident);
      deferred_words_ -= std::popcount(old.dirty & ~old.known);
    }
    table_[pos] = entry;
    if (entry.tag != kEmptyTag) {
      resident_words_ += std::popcount(entry.resident);
      deferred_words_ += std::popcount(entry.dirty & ~entry.known);
    }
  }

  void update_peaks() {
    m_.peak_resident_words = std::max(m_.peak_resident_words, resident_words_);
    m_.peak_deferred_words = std::max(m_.peak_deferred_words, deferred_words_);
  }
};

struct WordEntry {
  uint64_t tag = kEmptyTag;
  uint64_t home = 0;
  uint64_t last = 0;
  uint8_t offset = 0;
  bool known = false;
  bool dirty = false;
};

class FineStore final : public Store {
public:
  FineStore(Config config, Metrics &metrics)
      : Store(config, metrics), slots_(config.capacity_bytes / kWeightBytes),
        table_(slots_),
        line_positions_(
            ((uint64_t{kInputs} * kExcitatory + words_ - 1) / words_) * words_,
            -1) {
    if (slots_ == 0 || slots_ % words_ != 0)
      throw std::runtime_error("invalid fine capacity");
    if (c_.structure == Structure::SetFine) {
      ways_ = c_.parameter;
      const uint64_t physical_lines = slots_ / words_;
      if (physical_lines % ways_ != 0)
        throw std::runtime_error("fine capacity not divisible by ways");
      sets_ = physical_lines / ways_;
    } else if (c_.structure == Structure::HashWeight) {
      probe_limit_ = uint64_t{c_.parameter} * words_;
      blocks_ = slots_ / words_;
    } else {
      probe_limit_ = c_.parameter;
      bank_size_ = slots_ / words_;
    }
  }

  void access(const Event &event) override {
    count_access(event);
    ++clock_;
    const uint32_t inspected =
        event.kind == Kind::Continuous && c_.depression == Depression::Drain
            ? valid_mask(event.line, words_)
            : event.mask;
    auto [resident, known, dirty_before] = masks(event.line, inspected);
    if ((resident & event.mask) == event.mask)
      ++m_.hits;

    if (event.kind == Kind::Continuous && c_.depression == Depression::Drain) {
      if ((known & event.mask) != event.mask) {
        ++m_.reads;
        ++m_.materialization_reads;
      }
      write(resident | event.mask, (dirty_before & ~known) != 0,
            Kind::Continuous);
      invalidate_line(event.line);
      update_peaks();
      return;
    }

    const bool needs_read =
        event.kind == Kind::Continuous && (known & event.mask) != event.mask;
    if (event.kind == Kind::Sparse && c_.allocation == Allocation::Eager &&
        (known & event.mask) != event.mask) {
      ++m_.reads;
      allocate_from_read(event.line, event.mask, true);
    } else if (needs_read) {
      ++m_.reads;
      ++m_.materialization_reads;
      allocate_from_read(event.line, event.mask, true);
    }

    uint32_t todo = event.mask;
    while (todo) {
      const uint32_t offset = std::countr_zero(todo);
      todo &= todo - 1;
      auto pos = find(event.line, offset);
      if (!pos) {
        const bool known_value = event.kind == Kind::Continuous ||
                                 c_.allocation == Allocation::Eager;
        pos =
            insert_mandatory(make_entry(event.line, offset, known_value, true));
      }
      if (pos) {
        auto item = table_[*pos];
        if (event.kind == Kind::Continuous)
          item.known = true;
        item.dirty = true;
        item.last = clock_;
        assign_word(*pos, item);
      }
    }
    update_peaks();
  }

  void flush(bool final) override {
    std::vector<uint64_t> tags;
    tags.reserve(table_.size() / words_);
    for (const auto &item : table_) {
      if (item.tag != kEmptyTag)
        tags.push_back(item.tag);
    }
    std::sort(tags.begin(), tags.end());
    tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    for (uint64_t tag : tags)
      flush_line(tag, final);
    resident_words_ = deferred_words_ = 0;
  }

  uint64_t metadata_bytes() const override {
    return slots_ * (c_.structure == Structure::SetFine ? 17 : 10);
  }

private:
  uint64_t slots_;
  uint64_t ways_ = 1;
  uint64_t sets_ = 1;
  uint64_t blocks_ = 1;
  uint64_t bank_size_ = 1;
  uint64_t probe_limit_ = 1;
  uint64_t clock_ = 0;
  uint64_t resident_words_ = 0;
  uint64_t deferred_words_ = 0;
  std::vector<WordEntry> table_;
  // Simulator-side reverse locator for coordinated whole-line operations. This
  // is not part of the modeled storage or metadata and does not serve accesses.
  std::vector<int32_t> line_positions_;

  uint64_t set_for(uint64_t tag) const { return splitmix64(tag) % sets_; }

  uint64_t home_for(uint64_t tag, uint32_t offset) const {
    if (c_.structure == Structure::HashWeight) {
      return (splitmix64(tag) % blocks_) * words_ + offset;
    }
    if (c_.structure == Structure::HashOffset) {
      return uint64_t{offset} * bank_size_ + splitmix64(tag) % bank_size_;
    }
    return 0;
  }

  uint64_t next_position(uint64_t pos, uint32_t offset) const {
    if (c_.structure == Structure::HashWeight)
      return (pos + 1) % slots_;
    const uint64_t begin = uint64_t{offset} * bank_size_;
    return begin + ((pos - begin + 1) % bank_size_);
  }

  uint64_t distance(uint64_t home, uint64_t pos, uint32_t offset) const {
    if (c_.structure == Structure::HashWeight)
      return (pos + slots_ - home) % slots_;
    const uint64_t begin = uint64_t{offset} * bank_size_;
    const uint64_t h = home - begin;
    const uint64_t p = pos - begin;
    return (p + bank_size_ - h) % bank_size_;
  }

  WordEntry make_entry(uint64_t tag, uint32_t offset, bool known,
                       bool dirty) const {
    WordEntry item;
    item.tag = tag;
    item.offset = static_cast<uint8_t>(offset);
    item.home = home_for(tag, offset);
    item.last = clock_;
    item.known = known;
    item.dirty = dirty;
    return item;
  }

  std::optional<size_t> find(uint64_t tag, uint32_t offset) {
    if (c_.structure == Structure::SetFine) {
      const uint64_t set = set_for(tag);
      for (uint64_t way = 0; way < ways_; ++way) {
        ++m_.probes;
        const size_t pos = (set * ways_ + way) * words_ + offset;
        if (table_[pos].tag == tag)
          return pos;
      }
      m_.max_probe = std::max(m_.max_probe, ways_);
      return std::nullopt;
    }
    uint64_t pos = home_for(tag, offset);
    for (uint64_t probe = 0; probe < probe_limit_; ++probe) {
      ++m_.probes;
      if (table_[pos].tag == tag && table_[pos].offset == offset)
        return pos;
      pos = next_position(pos, offset);
    }
    m_.max_probe = std::max(m_.max_probe, probe_limit_);
    return std::nullopt;
  }

  std::optional<size_t> insert_mandatory(WordEntry item) {
    if (c_.structure == Structure::SetFine) {
      const uint64_t set = set_for(item.tag);
      size_t victim = (set * ways_) * words_ + item.offset;
      for (uint64_t way = 0; way < ways_; ++way) {
        const size_t pos = (set * ways_ + way) * words_ + item.offset;
        if (table_[pos].tag == kEmptyTag) {
          assign_word(pos, item);
          return pos;
        }
        if (table_[pos].last < table_[victim].last)
          victim = pos;
      }
      evict_line(table_[victim].tag, false, std::nullopt);
      for (uint64_t way = 0; way < ways_; ++way) {
        const size_t pos = (set * ways_ + way) * words_ + item.offset;
        if (table_[pos].tag == kEmptyTag) {
          assign_word(pos, item);
          return pos;
        }
      }
      throw std::logic_error(
          "coordinated set eviction did not free target lane");
    }

    const uint64_t target_tag = item.tag;
    const uint32_t target_offset = item.offset;
    uint64_t pos = item.home;
    uint64_t item_distance = 0;
    while (item_distance < probe_limit_) {
      ++m_.probes;
      if (table_[pos].tag == kEmptyTag) {
        assign_word(pos, item);
        return find(target_tag, target_offset);
      }
      const uint64_t occupant_distance =
          distance(table_[pos].home, pos, table_[pos].offset);
      if (occupant_distance < item_distance) {
        WordEntry displaced = table_[pos];
        assign_word(pos, item);
        item = displaced;
        item_distance = occupant_distance;
      }
      pos = next_position(pos, item.offset);
      ++item_distance;
    }
    ++m_.forced_evictions;
    evict_line(item.tag, true, item);
    return find(target_tag, target_offset);
  }

  void allocate_from_read(uint64_t tag, uint32_t mandatory_mask,
                          bool mark_known) {
    uint32_t mandatory = mandatory_mask;
    while (mandatory) {
      const uint32_t offset = std::countr_zero(mandatory);
      mandatory &= mandatory - 1;
      auto pos = find(tag, offset);
      if (!pos)
        pos = insert_mandatory(make_entry(tag, offset, mark_known, false));
      if (pos) {
        auto item = table_[*pos];
        item.known = mark_known;
        assign_word(*pos, item);
      }
    }
    uint32_t optional = valid_mask(tag, words_) & ~mandatory_mask;
    while (optional) {
      const uint32_t offset = std::countr_zero(optional);
      optional &= optional - 1;
      auto pos = find(tag, offset);
      if (pos) {
        auto item = table_[*pos];
        item.known = mark_known;
        assign_word(*pos, item);
        continue;
      }
      insert_if_empty(make_entry(tag, offset, mark_known, false));
    }
  }

  void insert_if_empty(const WordEntry &item) {
    if (c_.structure == Structure::SetFine) {
      const uint64_t set = set_for(item.tag);
      for (uint64_t way = 0; way < ways_; ++way) {
        const size_t pos = (set * ways_ + way) * words_ + item.offset;
        if (table_[pos].tag == kEmptyTag) {
          assign_word(pos, item);
          return;
        }
      }
      return;
    }
    uint64_t pos = item.home;
    for (uint64_t probe = 0; probe < probe_limit_; ++probe) {
      if (table_[pos].tag == kEmptyTag) {
        assign_word(pos, item);
        return;
      }
      pos = next_position(pos, item.offset);
    }
  }

  std::optional<size_t> locate(uint64_t tag, uint32_t offset) {
    const uint64_t key = tag * words_ + offset;
    if (key >= line_positions_.size())
      return std::nullopt;
    const int32_t pos = line_positions_[key];
    if (pos < 0)
      return std::nullopt;
    if (table_[pos].tag != tag || table_[pos].offset != offset)
      throw std::logic_error("fine reverse locator is inconsistent");
    return static_cast<size_t>(pos);
  }

  struct LineMasks {
    uint32_t resident;
    uint32_t known;
    uint32_t dirty;
  };

  template <typename Visitor> void visit_line(uint64_t tag, Visitor &&visitor) {
    uint32_t remaining = valid_mask(tag, words_);
    while (remaining) {
      const uint32_t offset = std::countr_zero(remaining);
      remaining &= remaining - 1;
      if (auto pos = locate(tag, offset))
        visitor(*pos);
    }
  }

  LineMasks masks(uint64_t tag, uint32_t requested = 0xffffffffU) {
    LineMasks result{0, 0, 0};
    const uint32_t valid = valid_mask(tag, words_);
    if ((requested & valid) == valid) {
      visit_line(tag, [&](size_t pos) {
        const auto &item = table_[pos];
        const uint32_t bit = uint32_t{1} << item.offset;
        result.resident |= bit;
        if (item.known)
          result.known |= bit;
        if (item.dirty)
          result.dirty |= bit;
      });
      return result;
    }
    uint32_t remaining = valid & requested;
    while (remaining) {
      const uint32_t offset = std::countr_zero(remaining);
      remaining &= remaining - 1;
      auto pos = locate(tag, offset);
      if (!pos)
        continue;
      const uint32_t bit = uint32_t{1} << offset;
      result.resident |= bit;
      if (table_[*pos].known)
        result.known |= bit;
      if (table_[*pos].dirty)
        result.dirty |= bit;
    }
    return result;
  }

  void invalidate_line(uint64_t tag) {
    visit_line(tag, [&](size_t pos) { assign_word(pos, WordEntry{}); });
  }

  void evict_line(uint64_t tag, bool already_counted_forced,
                  std::optional<WordEntry> external) {
    uint32_t resident = 0;
    uint32_t known = 0;
    uint32_t dirty = 0;
    visit_line(tag, [&](size_t pos) {
      const auto item = table_[pos];
      const uint32_t bit = uint32_t{1} << item.offset;
      resident |= bit;
      if (item.known)
        known |= bit;
      if (item.dirty)
        dirty |= bit;
      assign_word(pos, WordEntry{});
    });
    if (external) {
      const uint32_t bit = uint32_t{1} << external->offset;
      resident |= bit;
      if (external->known)
        known |= bit;
      if (external->dirty)
        dirty |= bit;
    }
    if (already_counted_forced) {
      // The Robin Hood insertion path increments the counter before calling.
    }
    if (dirty)
      write(resident, (dirty & ~known) != 0, Kind::Sparse);
  }

  void flush_line(uint64_t tag, bool final) {
    const auto [resident, known, dirty] = masks(tag);
    if (dirty) {
      ++m_.writebacks;
      m_.written_words += std::popcount(resident);
      if (dirty & ~known)
        ++m_.operator_writebacks;
      if (final)
        ++m_.final_writebacks;
      else
        ++m_.barrier_writebacks;
    }
    invalidate_line(tag);
  }

  void assign_word(size_t pos, const WordEntry &item) {
    const auto &old = table_[pos];
    if (old.tag != kEmptyTag) {
      line_positions_[old.tag * words_ + old.offset] = -1;
      --resident_words_;
      if (old.dirty && !old.known)
        --deferred_words_;
    }
    table_[pos] = item;
    if (item.tag != kEmptyTag) {
      line_positions_[item.tag * words_ + item.offset] =
          static_cast<int32_t>(pos);
      ++resident_words_;
      if (item.dirty && !item.known)
        ++deferred_words_;
    }
  }

  void update_peaks() {
    m_.peak_resident_words = std::max(m_.peak_resident_words, resident_words_);
    m_.peak_deferred_words = std::max(m_.peak_deferred_words, deferred_words_);
  }
};

std::unique_ptr<Store> make_store(const Config &config, Metrics &metrics) {
  if (config.structure == Structure::Direct)
    return std::make_unique<DirectStore>(config, metrics);
  if (config.structure == Structure::SetLine ||
      config.structure == Structure::HashLine)
    return std::make_unique<LineStore>(config, metrics);
  return std::make_unique<FineStore>(config, metrics);
}

struct Result {
  Config config;
  Metrics metrics;
  uint64_t metadata_bytes = 0;
  double simulation_seconds = 0;
};

Result simulate(const Config &config, const std::vector<Event> &events) {
  const auto start = Clock::now();
  Result result;
  result.config = config;
  auto store = make_store(config, result.metrics);
  result.metadata_bytes = store->metadata_bytes();
  for (const auto &event : events) {
    if (event.kind == Kind::Barrier) {
      if (config.flush_barriers)
        store->flush(false);
    } else {
      store->access(event);
    }
  }
  store->flush(true);
  result.simulation_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  return result;
}

void write_header(std::ostream &out) {
  out << "checkpoint,structure,allocation,depression,line_bytes,capacity_bytes,"
         "parameter,"
         "flush_barriers,input_spikes,excitatory_spikes,spike_records,attempts,"
         "event_count,"
         "parse_seconds,event_build_seconds,simulation_seconds,accesses,hits,"
         "logical_updates,"
         "sparse_updates,continuous_updates,reads,materialization_reads,"
         "writebacks,"
         "eviction_writebacks,drain_writebacks,barrier_writebacks,final_"
         "writebacks,"
         "operator_writebacks,written_words,forced_evictions,probes,max_probe,"
         "peak_resident_words,peak_deferred_words,metadata_bytes\n";
}

void write_result(std::ostream &out, const Trace &trace, size_t event_count,
                  double parse_seconds, double build_seconds,
                  const Result &result) {
  const auto &c = result.config;
  const auto &m = result.metrics;
  out << trace.checkpoint << ',' << name(c.structure) << ','
      << name(c.allocation) << ',' << name(c.depression) << ',' << c.line_bytes
      << ',' << c.capacity_bytes << ',' << c.parameter << ','
      << (c.flush_barriers ? 1 : 0) << ',' << trace.input_spikes << ','
      << trace.excitatory_spikes << ',' << trace.spike_records << ','
      << trace.attempts << ',' << event_count << ',' << std::setprecision(9)
      << parse_seconds << ',' << build_seconds << ','
      << result.simulation_seconds << ',' << m.accesses << ',' << m.hits << ','
      << m.logical_updates << ',' << m.sparse_updates << ','
      << m.continuous_updates << ',' << m.reads << ','
      << m.materialization_reads << ',' << m.writebacks << ','
      << m.eviction_writebacks << ',' << m.drain_writebacks << ','
      << m.barrier_writebacks << ',' << m.final_writebacks << ','
      << m.operator_writebacks << ',' << m.written_words << ','
      << m.forced_evictions << ',' << m.probes << ',' << m.max_probe << ','
      << m.peak_resident_words << ',' << m.peak_deferred_words << ','
      << result.metadata_bytes << '\n';
}

std::vector<Config> matrix_configs(uint32_t line_bytes) {
  std::vector<Config> configs;
  for (auto allocation : {Allocation::Eager, Allocation::Deferred}) {
    for (auto depression : {Depression::WriteBack, Depression::Drain}) {
      configs.push_back(
          {Structure::Direct, allocation, depression, line_bytes, 0, 0, true});
    }
  }
  for (uint64_t capacity :
       {128 * kKiB, 256 * kKiB, 512 * kKiB, kMiB, 2 * kMiB, 4 * kMiB}) {
    for (auto allocation : {Allocation::Eager, Allocation::Deferred}) {
      for (auto depression : {Depression::WriteBack, Depression::Drain}) {
        for (uint32_t ways : {1U, 2U, 4U, 8U}) {
          configs.push_back({Structure::SetLine, allocation, depression,
                             line_bytes, capacity, ways, true});
          configs.push_back({Structure::SetFine, allocation, depression,
                             line_bytes, capacity, ways, true});
        }
        for (uint32_t distance : {1U, 2U, 4U, 8U}) {
          configs.push_back({Structure::HashLine, allocation, depression,
                             line_bytes, capacity, distance, true});
          configs.push_back({Structure::HashWeight, allocation, depression,
                             line_bytes, capacity, distance, true});
          configs.push_back({Structure::HashOffset, allocation, depression,
                             line_bytes, capacity, distance, true});
        }
      }
    }
  }
  return configs;
}

void self_test() {
  if (matrix_configs(64).size() != 484)
    throw std::runtime_error("expanded matrix configuration count failed");
  {
    const uint32_t words = 16;
    auto mask = valid_mask((uint64_t{kInputs} * kExcitatory) / words, words);
    if (mask != 0)
      throw std::runtime_error("valid-mask end test failed");
  }
  std::vector<Event> events = {
      {Kind::Sparse, 3, 1U << 2},
      {Kind::Sparse, 3, 1U << 5},
      {Kind::Continuous, 3, 0xffU},
      {Kind::Barrier, 0, 0},
  };
  for (auto structure :
       {Structure::Direct, Structure::SetLine, Structure::SetFine,
        Structure::HashLine, Structure::HashWeight, Structure::HashOffset}) {
    Config c{structure, Allocation::Deferred, Depression::Drain, 32, 256, 2,
             true};
    auto result = simulate(c, events);
    if (result.metrics.logical_updates != 10)
      throw std::runtime_error("logical update count failed");
    if (result.metrics.writebacks == 0)
      throw std::runtime_error("writeback test failed for " + name(structure));
    if (structure == Structure::Direct &&
        (result.metrics.reads != 8 || result.metrics.writebacks != 10 ||
         result.metrics.operator_writebacks != 2 ||
         result.metrics.written_words != 10))
      throw std::runtime_error("direct per-weight accounting failed");
    if (structure == Structure::HashWeight && result.metrics.max_probe != 16)
      throw std::runtime_error("individual-weight probe scaling failed");
  }
  {
    Config c{Structure::SetFine,
             Allocation::Deferred,
             Depression::Drain,
             32,
             256,
             2,
             true};
    auto result = simulate(c, events);
    if (result.metrics.writebacks != 1 ||
        result.metrics.drain_writebacks != 1 ||
        result.metrics.written_words != 8)
      throw std::runtime_error("fine-grained coordinated drain failed");
  }
  {
    uint64_t first = 0;
    uint64_t second = 1;
    while (splitmix64(first) % 2 != splitmix64(second) % 2)
      ++second;
    Config c{Structure::SetFine,
             Allocation::Deferred,
             Depression::WriteBack,
             32,
             64,
             1,
             true};
    std::vector<Event> conflict = {
        {Kind::Sparse, static_cast<uint32_t>(first), 0b11},
        {Kind::Sparse, static_cast<uint32_t>(second), 0b01},
        {Kind::Barrier, 0, 0}};
    auto result = simulate(c, conflict);
    if (result.metrics.writebacks != 2 ||
        result.metrics.eviction_writebacks != 1 ||
        result.metrics.barrier_writebacks != 1 ||
        result.metrics.written_words != 3)
      throw std::runtime_error("fine-grained coordinated eviction failed");
  }
  {
    Config c{Structure::SetLine,
             Allocation::Deferred,
             Depression::WriteBack,
             32,
             32,
             1,
             true};
    std::vector<Event> conflict = {
        {Kind::Sparse, 0, 1}, {Kind::Sparse, 1, 1}, {Kind::Barrier, 0, 0}};
    auto result = simulate(c, conflict);
    if (result.metrics.writebacks != 2 ||
        result.metrics.operator_writebacks != 2)
      throw std::runtime_error("set-line eviction/flush test failed");
  }
  {
    Trace trace;
    trace.checkpoint = 1;
    trace.input_spikes = 1;
    trace.excitatory_spikes = 1;
    trace.attempts = 1;
    trace.barriers = {0};
    trace.ticks.push_back({{0}, {0}});
    const auto events = build_events(trace, 64);
    if (events.size() != 810)
      throw std::runtime_error("synthetic event count failed");
    Config c{Structure::SetLine,
             Allocation::Eager,
             Depression::WriteBack,
             64,
             4096,
             2,
             true};
    auto result = simulate(c, events);
    if (result.metrics.accesses != 809 ||
        result.metrics.logical_updates != 1184 ||
        result.metrics.sparse_updates != 784 ||
        result.metrics.continuous_updates != 400) {
      throw std::runtime_error("synthetic trace reconstruction failed");
    }
  }
  std::cout << "self-test: passed\n";
}

struct Arguments {
  std::string command;
  std::string trace;
  std::string output;
  Config config;
  bool e_before_x = false;
};

Arguments parse_arguments(int argc, char **argv) {
  if (argc < 2)
    throw std::runtime_error(
        "usage: trace-sim <run|matrix|--self-test> [options]");
  Arguments args;
  args.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc)
        throw std::runtime_error("missing value for " + key);
      return argv[i];
    };
    if (key == "--trace")
      args.trace = value();
    else if (key == "--output")
      args.output = value();
    else if (key == "--structure")
      args.config.structure = parse_structure(value());
    else if (key == "--line-size")
      args.config.line_bytes = std::stoul(value());
    else if (key == "--capacity")
      args.config.capacity_bytes = parse_capacity(value());
    else if (key == "--ways" || key == "--distance")
      args.config.parameter = std::stoul(value());
    else if (key == "--allocation")
      args.config.allocation = parse_allocation(value());
    else if (key == "--depression")
      args.config.depression = parse_depression(value());
    else if (key == "--barriers") {
      const auto setting = value();
      if (setting != "flush" && setting != "none")
        throw std::runtime_error("barriers must be flush or none");
      args.config.flush_barriers = setting == "flush";
    } else if (key == "--event-order") {
      const auto order = value();
      if (order != "x-e" && order != "e-x")
        throw std::runtime_error("event order must be x-e or e-x");
      args.e_before_x = order == "e-x";
    } else
      throw std::runtime_error("unknown option: " + key);
  }
  if (args.command != "--self-test" &&
      (args.trace.empty() || args.output.empty()))
    throw std::runtime_error("--trace and --output are required");
  if (args.config.line_bytes != 32 && args.config.line_bytes != 64 &&
      args.config.line_bytes != 128)
    throw std::runtime_error("line size must be 32, 64, or 128");
  return args;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto args = parse_arguments(argc, argv);
    if (args.command == "--self-test") {
      self_test();
      return 0;
    }
    const auto parse_start = Clock::now();
    const auto trace = read_trace(args.trace);
    const double parse_seconds =
        std::chrono::duration<double>(Clock::now() - parse_start).count();
    std::ofstream output(args.output);
    if (!output)
      throw std::runtime_error("cannot create output: " + args.output);
    write_header(output);
    if (args.command == "run") {
      const auto build_start = Clock::now();
      const auto events =
          build_events(trace, args.config.line_bytes, args.e_before_x);
      const double build_seconds =
          std::chrono::duration<double>(Clock::now() - build_start).count();
      const auto result = simulate(args.config, events);
      write_result(output, trace, events.size(), parse_seconds, build_seconds,
                   result);
      std::cerr << "simulated " << events.size() << " events in "
                << result.simulation_seconds << " s\n";
    } else if (args.command == "matrix") {
      const auto matrix_start = Clock::now();
      size_t count = 0;
      for (uint32_t line_bytes : {32U, 64U, 128U}) {
        const auto build_start = Clock::now();
        const auto events = build_events(trace, line_bytes, args.e_before_x);
        const double build_seconds =
            std::chrono::duration<double>(Clock::now() - build_start).count();
        for (const auto &config : matrix_configs(line_bytes)) {
          const auto result = simulate(config, events);
          write_result(output, trace, events.size(), parse_seconds,
                       build_seconds, result);
          ++count;
        }
      }
      const double total =
          std::chrono::duration<double>(Clock::now() - matrix_start).count();
      std::cerr << "completed " << count << " configurations in " << total
                << " s\n";
    } else {
      throw std::runtime_error("unknown command: " + args.command);
    }
  } catch (const std::exception &error) {
    std::cerr << "trace-sim: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
