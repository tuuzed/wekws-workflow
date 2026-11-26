#include "engine/json.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace haibingtech {
namespace kws {

bool Json::has(const std::string& key) const {
  return type_ == Type::Object && object_.find(key) != object_.end();
}

const Json& Json::at(const std::string& key) const {
  if (type_ != Type::Object) {
    throw std::runtime_error("Json::at called on non-object");
  }
  auto it = object_.find(key);
  if (it == object_.end()) {
    throw std::runtime_error("Json key not found: " + key);
  }
  return it->second;
}

bool Json::get_bool(const std::string& key, bool default_value) const {
  if (!has(key) || !at(key).is_bool()) return default_value;
  return at(key).boolean();
}

int Json::get_int(const std::string& key, int default_value) const {
  if (!has(key) || !at(key).is_number()) return default_value;
  return static_cast<int>(at(key).number());
}

double Json::get_double(const std::string& key, double default_value) const {
  if (!has(key) || !at(key).is_number()) return default_value;
  return at(key).number();
}

std::string Json::get_string(const std::string& key,
                             const std::string& default_value) const {
  if (!has(key) || !at(key).is_string()) return default_value;
  return at(key).string();
}

namespace {

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text) {}

  Json Parse() {
    SkipWhitespace();
    Json value = ParseValue();
    SkipWhitespace();
    if (pos_ != text_.size()) {
      throw std::runtime_error("Invalid JSON: trailing characters");
    }
    return value;
  }

 private:
  void SkipWhitespace() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  char Peek() const {
    if (pos_ >= text_.size()) throw std::runtime_error("Unexpected end of JSON");
    return text_[pos_];
  }

  void Expect(char c) {
    if (Peek() != c) {
      throw std::runtime_error(std::string("Expected '") + c + "' in JSON");
    }
    ++pos_;
  }

  Json ParseValue() {
    SkipWhitespace();
    char c = Peek();
    if (c == '{') return ParseObject();
    if (c == '[') return ParseArray();
    if (c == '"') {
      Json value;
      value.type_ = Json::Type::String;
      value.string_ = ParseString();
      return value;
    }
    if (c == 't') {
      ConsumeLiteral("true");
      Json value;
      value.type_ = Json::Type::Bool;
      value.bool_ = true;
      return value;
    }
    if (c == 'f') {
      ConsumeLiteral("false");
      Json value;
      value.type_ = Json::Type::Bool;
      value.bool_ = false;
      return value;
    }
    if (c == 'n') {
      ConsumeLiteral("null");
      return Json();
    }
    return ParseNumber();
  }

  void ConsumeLiteral(const char* literal) {
    for (const char* p = literal; *p != '\0'; ++p) {
      if (pos_ >= text_.size() || text_[pos_] != *p) {
        throw std::runtime_error("Invalid JSON literal");
      }
      ++pos_;
    }
  }

  Json ParseNumber() {
    size_t begin = pos_;
    if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) {
      ++pos_;
    }
    while (pos_ < text_.size() &&
           (std::isdigit(static_cast<unsigned char>(text_[pos_])) ||
            text_[pos_] == '.' || text_[pos_] == 'e' || text_[pos_] == 'E' ||
            text_[pos_] == '-' || text_[pos_] == '+')) {
      ++pos_;
    }
    if (begin == pos_) throw std::runtime_error("Invalid JSON number");
    Json value;
    value.type_ = Json::Type::Number;
    value.number_ = std::stod(text_.substr(begin, pos_ - begin));
    return value;
  }

  static void AppendUtf8(std::string* out, uint32_t codepoint) {
    if (codepoint <= 0x7f) {
      out->push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
      out->push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
      out->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
      out->push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
      out->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
      out->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
      out->push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
      out->push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
      out->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
      out->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
  }

  uint32_t ParseHex4() {
    if (pos_ + 4 > text_.size()) {
      throw std::runtime_error("Invalid JSON unicode escape");
    }
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      char c = text_[pos_++];
      value <<= 4;
      if (c >= '0' && c <= '9') value |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') value |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') value |= static_cast<uint32_t>(c - 'A' + 10);
      else throw std::runtime_error("Invalid JSON unicode escape");
    }
    return value;
  }

  std::string ParseString() {
    Expect('"');
    std::string result;
    while (pos_ < text_.size()) {
      char c = text_[pos_++];
      if (c == '"') return result;
      if (c != '\\') {
        result.push_back(c);
        continue;
      }
      if (pos_ >= text_.size()) throw std::runtime_error("Invalid JSON escape");
      char esc = text_[pos_++];
      switch (esc) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        case 'u': {
          uint32_t cp = ParseHex4();
          if (cp >= 0xd800 && cp <= 0xdbff && pos_ + 6 <= text_.size() &&
              text_[pos_] == '\\' && text_[pos_ + 1] == 'u') {
            pos_ += 2;
            uint32_t low = ParseHex4();
            cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
          }
          AppendUtf8(&result, cp);
          break;
        }
        default: throw std::runtime_error("Invalid JSON escape");
      }
    }
    throw std::runtime_error("Unterminated JSON string");
  }

  Json ParseArray() {
    Expect('[');
    Json value;
    value.type_ = Json::Type::Array;
    SkipWhitespace();
    if (Peek() == ']') {
      ++pos_;
      return value;
    }
    while (true) {
      value.array_.push_back(ParseValue());
      SkipWhitespace();
      char c = Peek();
      if (c == ']') {
        ++pos_;
        return value;
      }
      Expect(',');
    }
  }

  Json ParseObject() {
    Expect('{');
    Json value;
    value.type_ = Json::Type::Object;
    SkipWhitespace();
    if (Peek() == '}') {
      ++pos_;
      return value;
    }
    while (true) {
      SkipWhitespace();
      std::string key = ParseString();
      SkipWhitespace();
      Expect(':');
      value.object_[key] = ParseValue();
      SkipWhitespace();
      char c = Peek();
      if (c == '}') {
        ++pos_;
        return value;
      }
      Expect(',');
    }
  }

  const std::string& text_;
  size_t pos_ = 0;
};

}  // namespace

Json ParseJson(const std::string& text) {
  return JsonParser(text).Parse();
}

}  // namespace kws
}  // namespace haibingtech
