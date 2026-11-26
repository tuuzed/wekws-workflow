#ifndef HAIBINGTECH_KWS_JSON_H_
#define HAIBINGTECH_KWS_JSON_H_

#include <map>
#include <string>
#include <vector>

namespace haibingtech {
namespace kws {

class Json {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  bool boolean() const { return bool_; }
  double number() const { return number_; }
  const std::string& string() const { return string_; }
  const std::vector<Json>& array() const { return array_; }
  const std::map<std::string, Json>& object() const { return object_; }

  bool has(const std::string& key) const;
  const Json& at(const std::string& key) const;

  // Convenience accessors with defaults.
  bool get_bool(const std::string& key, bool default_value) const;
  int get_int(const std::string& key, int default_value) const;
  double get_double(const std::string& key, double default_value) const;
  std::string get_string(const std::string& key,
                         const std::string& default_value) const;

 public:
  friend Json ParseJson(const std::string& text);

  Type type_ = Type::Null;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<Json> array_;
  std::map<std::string, Json> object_;
};

Json ParseJson(const std::string& text);

}  // namespace kws
}  // namespace haibingtech

#endif  // HAIBINGTECH_KWS_JSON_H_
