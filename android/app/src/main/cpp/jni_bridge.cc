#include <jni.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kws_engine.h"
#include "engine/runtime_config.h"

namespace {

void ThrowJava(JNIEnv* env, const char* message) {
  jclass exception_class = env->FindClass("java/lang/RuntimeException");
  if (exception_class != nullptr) {
    env->ThrowNew(exception_class, message);
  }
}

std::string ToStdString(JNIEnv* env, jstring value) {
  if (value == nullptr) return "";
  const char* chars = env->GetStringUTFChars(value, nullptr);
  std::string result = chars == nullptr ? "" : std::string(chars);
  if (chars != nullptr) env->ReleaseStringUTFChars(value, chars);
  return result;
}

haibingtech::kws::KwsEngine* GetEngine(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    ThrowJava(env, "KWS native handle is 0");
    return nullptr;
  }
  return reinterpret_cast<haibingtech::kws::KwsEngine*>(handle);
}

std::string JsonEscape(const std::string& input) {
  std::string output;
  output.reserve(input.size() + 8);
  for (unsigned char c : input) {
    switch (c) {
      case '"': output += "\\\""; break;
      case '\\': output += "\\\\"; break;
      case '\b': output += "\\b"; break;
      case '\f': output += "\\f"; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      default:
        if (c < 0x20) {
          static const char* kHex = "0123456789abcdef";
          output += "\\u00";
          output.push_back(kHex[(c >> 4) & 0xf]);
          output.push_back(kHex[c & 0xf]);
        } else {
          output.push_back(static_cast<char>(c));
        }
    }
  }
  return output;
}

void AppendTokenProbJson(std::ostream& oss,
                         const std::vector<haibingtech::kws::TokenProb>& items) {
  oss << "[";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) oss << ",";
    const auto& item = items[i];
    oss << "{\"token\":\"" << JsonEscape(item.token) << "\""
        << ",\"token_id\":" << item.token_id
        << ",\"frame\":" << item.frame
        << ",\"prob\":" << std::fixed << std::setprecision(6) << item.prob
        << "}";
  }
  oss << "]";
}

void AppendFrameTokenProbJson(
    std::ostream& oss,
    const std::vector<haibingtech::kws::FrameTokenProb>& items) {
  oss << "[";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) oss << ",";
    const auto& item = items[i];
    oss << "{\"token_id\":" << item.token_id
        << ",\"prob\":" << std::fixed << std::setprecision(6) << item.prob
        << "}";
  }
  oss << "]";
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_haibingtech_kws_KwsJni_create(
    JNIEnv* env, jclass, jstring model_path, jstring config_json, jint channels) {
  try {
    haibingtech::kws::RuntimeConfig config =
        haibingtech::kws::ParseRuntimeConfig(ToStdString(env, config_json));
    auto* engine = new haibingtech::kws::KwsEngine(
        ToStdString(env, model_path), config, static_cast<int>(channels));
    return reinterpret_cast<jlong>(engine);
  } catch (const std::exception& e) {
    ThrowJava(env, e.what());
    return 0;
  }
}

JNIEXPORT void JNICALL Java_com_haibingtech_kws_KwsJni_feed(
    JNIEnv* env, jclass, jlong handle, jint channel, jshortArray pcm,
    jint offset, jint length) {
  try {
    haibingtech::kws::KwsEngine* engine = GetEngine(env, handle);
    if (engine == nullptr || pcm == nullptr) return;
    const jsize total = env->GetArrayLength(pcm);
    if (offset < 0 || length < 0 || offset + length > total) {
      throw std::out_of_range("invalid pcm offset or length");
    }
    std::vector<int16_t> samples(static_cast<size_t>(length));
    if (length > 0) {
      env->GetShortArrayRegion(pcm, offset, length,
                               reinterpret_cast<jshort*>(samples.data()));
    }
    engine->AcceptWaveform(static_cast<int>(channel), samples.data(),
                           samples.size());
  } catch (const std::exception& e) {
    ThrowJava(env, e.what());
  }
}

JNIEXPORT void JNICALL Java_com_haibingtech_kws_KwsJni_finish(
    JNIEnv* env, jclass, jlong handle, jint channel) {
  try {
    haibingtech::kws::KwsEngine* engine = GetEngine(env, handle);
    if (engine != nullptr) engine->Finish(static_cast<int>(channel));
  } catch (const std::exception& e) {
    ThrowJava(env, e.what());
  }
}

JNIEXPORT jboolean JNICALL Java_com_haibingtech_kws_KwsJni_decodeOnce(
    JNIEnv* env, jclass, jlong handle) {
  try {
    haibingtech::kws::KwsEngine* engine = GetEngine(env, handle);
    if (engine == nullptr) return JNI_FALSE;
    return engine->DecodeOnce() ? JNI_TRUE : JNI_FALSE;
  } catch (const std::exception& e) {
    ThrowJava(env, e.what());
    return JNI_FALSE;
  }
}

JNIEXPORT jstring JNICALL Java_com_haibingtech_kws_KwsJni_currentEventsJson(
    JNIEnv* env, jclass, jlong handle) {
  try {
    haibingtech::kws::KwsEngine* engine = GetEngine(env, handle);
    if (engine == nullptr) return env->NewStringUTF("[]");

    std::vector<haibingtech::kws::DetectionEvent> events =
        engine->CurrentEvents();
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < events.size(); ++i) {
      if (i > 0) oss << ",";
      const auto& event = events[i];
      oss << "{\"channel\":" << event.channel
          << ",\"keyword\":\"" << JsonEscape(event.keyword) << "\""
          << ",\"score\":" << std::fixed << std::setprecision(6) << event.score
          << ",\"start_ms\":" << event.start_frame * engine->frame_shift_ms()
          << ",\"end_ms\":" << event.end_frame * engine->frame_shift_ms()
          << ",\"token_probs\":";
      AppendTokenProbJson(oss, event.token_probs);
      oss << "}";
    }
    oss << "]";
    return env->NewStringUTF(oss.str().c_str());
  } catch (const std::exception& e) {
    ThrowJava(env, e.what());
    return env->NewStringUTF("[]");
  }
}

JNIEXPORT jstring JNICALL Java_com_haibingtech_kws_KwsJni_frameDebugJson(
    JNIEnv* env, jclass, jlong handle, jint channel, jint max_frames) {
  try {
    haibingtech::kws::KwsEngine* engine = GetEngine(env, handle);
    if (engine == nullptr) return env->NewStringUTF("[]");

    std::vector<haibingtech::kws::FrameDebug> frames = engine->FrameDebugDump(
        static_cast<int>(channel), static_cast<size_t>(std::max(0, max_frames)));
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < frames.size(); ++i) {
      if (i > 0) oss << ",";
      const auto& frame = frames[i];
      oss << "{\"frame\":" << frame.frame
          << ",\"time_ms\":" << frame.frame * engine->frame_shift_ms()
          << ",\"raw_topk\":";
      AppendFrameTokenProbJson(oss, frame.raw_topk);
      oss << ",\"kept\":";
      AppendFrameTokenProbJson(oss, frame.kept);
      oss << "}";
    }
    oss << "]";
    return env->NewStringUTF(oss.str().c_str());
  } catch (const std::exception& e) {
    ThrowJava(env, e.what());
    return env->NewStringUTF("[]");
  }
}

JNIEXPORT void JNICALL Java_com_haibingtech_kws_KwsJni_resetChannel(
    JNIEnv* env, jclass, jlong handle, jint channel) {
  try {
    haibingtech::kws::KwsEngine* engine = GetEngine(env, handle);
    if (engine != nullptr) engine->ResetChannel(static_cast<int>(channel));
  } catch (const std::exception& e) {
    ThrowJava(env, e.what());
  }
}

JNIEXPORT void JNICALL Java_com_haibingtech_kws_KwsJni_destroy(
    JNIEnv*, jclass, jlong handle) {
  delete reinterpret_cast<haibingtech::kws::KwsEngine*>(handle);
}

}  // extern "C"
