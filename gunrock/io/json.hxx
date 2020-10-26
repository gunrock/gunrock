/**
 * @file
 * json.hxx
 *
 * @brief json structure to build json with application specific stats.
 */

#pragma once

#include <cmath>
#include <cstdio>
#include <ctime>
#include <time.h>
#include <vector>

// RapidJSON includes (required)
#include <rapidjson/document.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>

#include <gunrock/error.hxx>
#include <gunrock/util/gitsha1.hxx>

namespace gunrock {
namespace io {

/**
 * @brief json data structure to build output a json to stdout or a file.
 */
class json {
  std::string _time_str;
  std::string _application_name;

  // json filename and file (rw, open)
  std::string _filename;
  std::FILE* _file;

  // RapidJSON setup for writing json
  typedef rapidjson::StringBuffer buffer_t;
  typedef rapidjson::PrettyWriter<rapidjson::StringBuffer> writer_t;
  typedef rapidjson::Document document_t;
  typedef rapidjson::Value value_t;

  buffer_t* _stream;
  writer_t* _writer;
  document_t* _document;

 public:
  /**
   * @brief info default constructor
   */
  info()
      : _time_str(""),
        _application_name(""),
        _filename(""),
        _stream(nullptr),
        _writer(nullptr),
        _file(nullptr),
        _document(nullptr) {}

  info(std::string application, std::string filename)
      : _time_str(""),
        _application_name(application),
        _filename(filename),
        _stream(nullptr),
        _writer(nullptr),
        _file(nullptr),
        _document(nullptr) {
    init();
  }

  ~info() {
    _time_str = "";
    _application_name = "";
    _filename = "";

    delete _stream;
    _stream = nullptr;

    delete _writer;
    _writer = nullptr;

    delete _document;
    _document = nullptr;
  }

  /**
   * @brief Initialization process for info.
   *
   * @param[in] application_name application name.
   * @param[in] args Command line arguments.
   */
  void init_base(std::string application_name) {
    // To keep things uniform, convert application name to lower case
    std::transform(application_name.begin(), application_name.end(),
                   application_name.begin(), ::tolower);

    this->_application_name = application_name;

    // Get time (with milliseconds accuracy). We use time with ms as a filename
    // identifier to keep every run named differently. Milliseconds accuracy is
    // important because some runs may execute within a single second and will
    // overwrite each other as a single filename.
    time_t now = time(NULL);

    long ms;   // Milliseconds
    time_t s;  // Seconds
    struct timespec spec;

    clock_gettime(CLOCK_REALTIME, &spec);

    s = spec.tv_sec;
    ms = round(spec.tv_nsec / 1.0e6);  // Convert nanoseconds to milliseconds
    if (ms > 999) {
      ++s;
      ms = 0;
    }

    std::string time_s = std::string(ctime(&now));
    std::string time_ms = std::to_string(ms);

    this->_time_str = time_s;  // Used within json
    std::string time_str_filename = time_s.substr(0, time_s.size() - 5) +
                                    time_ms + ' ' +
                                    time_s.substr(time_s.length() - 5);
  }

  /**
   * @brief Initialization process for Info.
   */
  void init() {
    init_base(_application_name);
    _document = new document_t();
    _document->SetObject();

    // Use a StringBuffer to hold the file data. This requires we manually
    // fputs the data from the stream to a file, but it's unlikely we would
    // ever write incomplete files. With a FileWriteStream, rapidjson will
    // decide to start writing whenever the buffer we provide is full
    _stream = new buffer_t();
    _writer = new writer_t(*stream);

    // Write the initial copy of the file with an invalid json-integrity
    /*
     * @todo ... SetBaseInfo(_application_name, false);
     */

    // Traverse the document for writing events
    if (stream != NULL) {
      _document->Accept(*writer);
      assert(writer->IsComplete());
    }
    if (_filename != "") {
      _file = std::fopen(_filename.c_str(), "w");
      std::fputs(_stream->GetString(), _file);
      std::fclose(_file);
    }

    // We now start over with a new stream and writer. We can't reuse them.
    // We also reset the document and rewrite our initial data - this time
    // without the invalid time.
    delete _stream;
    delete _writer;

    _stream = new buffer_t();
    _writer = new writer_t(*_stream);

    _document->SetObject();
    /*
     * @todo do stuff...
     */
  }

  // Get values from the json document.
  // get <start>
  template <typename T>
  value_t get_val(const T& val) {
    return value_t(val);
  }

  value_t get_val(const std::string& str) {
    if (_document == NULL)
      return value_t();
    else {
      return value_t(str, _document->GetAllocator());
    }
  }

  value_t get_val(char* const str) {
    if (_document == NULL)
      return value_t();
    else {
      return value_t(str, _document->GetAllocator());
    }
  }
  // get <end>

  // Set values and type support.
  // set <start>
  template <typename T>
  void set_bool(std::string name, const T& val, value_t& json_object) {
    std::cerr << "Attempt to set_val with unknown type for key \"" << name
              << std::endl;
  }

  void set_bool(std::string name, const bool& val, value_t& json_object) {
    if (_document != NULL) {
      value_t key(name, _document->GetAllocator());
      json_object.AddMember(key, val, _document->GetAllocator());
    }
  }

  template <typename T>
  void set_int(std::string name, const T& val, value_t& json_object) {
    std::cerr << "Attempt to set_val with unknown type for key \"" << name
              << std::endl;
  }

  void set_int(std::string name, const int& val, value_t& json_object) {
    if (_document != NULL) {
      value_t key(name, _document->GetAllocator());
      json_object.AddMember(key, val, _document->GetAllocator());
    }
  }

  template <typename T>
  void set_uint(std::string name, const T& val, value_t& json_object) {
    std::cerr << "Attempt to set_val with unknown type for key \"" << name
              << std::endl;
  }

  void set_uint(std::string name,
                const unsigned int& val,
                value_t& json_object) {
    if (_document != NULL) {
      value_t key(name, _document->GetAllocator());
      json_object.AddMember(key, val, _document->GetAllocator());
    }
  }

  template <typename T>
  void set_int64(std::string name, const T& val, value_t& json_object) {
    std::cerr << "writing unknown type for key \"" << name << std::endl;
  }

  void set_int64(std::string name, const int64_t& val, value_t& json_object) {
    if (_document != NULL) {
      value_t key(name, _document->GetAllocator());
      json_object.AddMember(key, val, _document->GetAllocator());
    }
  }

  template <typename T>
  void set_uint64(std::string name, const T& val, value_t& json_object) {
    std::cerr << "writing unknown type for key \"" << name << std::endl;
  }

  void set_uint64(std::string name, const uint64_t& val, value_t& json_object) {
    if (_document != NULL) {
      value_t key(name, _document->GetAllocator());
      json_object.AddMember(key, val, _document->GetAllocator());
    }
  }

  template <typename T>
  void set_double(std::string name, const T& val, value_t& json_object) {
    std::cerr << "writing unknown type for key \"" << name << std::endl;
  }

  void set_double(std::string name, const float& val, value_t& json_object) {
    if (_document != NULL) {
      value_t key(name, _document->GetAllocator());

      // Doubles and floats have an edge case. INF and NAN are valid values for
      // a double, but JSON doesn't allow them in the official spec. Some json
      // formats still allow them. We have to choose a behavior here, so let's
      // output the value as a string
      if (std::isinf(val) || std::isnan(val)) {
        value_t null_val(rapidjson::kNullType);
        json_object.AddMember(key, null_val, _document->GetAllocator());
      } else {
        json_object.AddMember(key, val, _document->GetAllocator());
      }
    }
  }

  void set_double(std::string name, const double& val, value_t& json_object) {
    if (_document != NULL) {
      value_t key(name, _document->GetAllocator());

      if (std::isinf(val) || std::isnan(val)) {
        value_t null_val(rapidjson::kNullType);
        json_object.AddMember(key, null_val, _document->GetAllocator());
      } else {
        json_object.AddMember(key, val, _document->GetAllocator());
      }
    }
  }

  // Attach a key with name, "name" and value "val" to the JSON object
  // "json_object"
  template <typename T>
  void set_val(std::string name, const T& val, value_t& json_object) {
    if (_document == NULL)
      return;

    auto tidx = std::type_index(typeid(T));

    // TODO: Use constexpr if for this instead of the runtime check
    // Then we won't need the filler functions above for a generic
    // template parameter T
    if (tidx == std::type_index(typeid(bool)))
      set_bool(name, val, json_object);
    else if (tidx == std::type_index(typeid(char)) ||
             tidx == std::type_index(typeid(signed char)) ||
             tidx == std::type_index(typeid(short)) ||
             tidx == std::type_index(typeid(int)))
      set_int(name, val, json_object);
    else if (tidx == std::type_index(typeid(unsigned char)) ||
             tidx == std::type_index(typeid(unsigned short)) ||
             tidx == std::type_index(typeid(unsigned int)))
      set_uint(name, val, json_object);
    else if (tidx == std::type_index(typeid(long)) ||
             tidx == std::type_index(typeid(long long)))
      set_int64(name, val, json_object);
    else if (tidx == std::type_index(typeid(unsigned long)) ||
             tidx == std::type_index(typeid(unsigned long long)))
      set_uint64(name, val, json_object);
    else if (tidx == std::type_index(typeid(float)) ||
             tidx == std::type_index(typeid(double)) ||
             tidx == std::type_index(typeid(long double)))
      set_double(name, val, json_object);
    else {
      std::ostringstream ostr;
      ostr << val;
      std::string str = ostr.str();

      value_t key(name, _document->GetAllocator());
      json_object.AddMember(key, str, _document->GetAllocator());
    }
  }

  template <typename T>
  void set_val(std::string name,
               const std::vector<T>& vec,
               value_t& json_object) {
    /*
     * @todo update parameters to support "ALWAYS_ARRAY" type currently using a
     * hack to make sure tag is always an array in JSON. This is also required
     * for fields such as srcs, process-times, etc.
     */
    if (_document == NULL)
      return;

    if (vec.size() == 1 && (name.compare("tag") != 0)) {
      set_val(name, vec.front(), json_object);
    } else {
      value_t arr(rapidjson::kArrayType);
      for (const T& i : vec) {
        value_t val = get_val(i);

        arr.PushBack(val, _document->GetAllocator());
      }

      value_t key(name, _document->GetAllocator());
      json_object.AddMember(key, arr, _document->GetAllocator());
    }
  }

  template <typename T>
  void set_val(std::string name,
               const std::vector<std::pair<T, T>>& vec,
               value_t& json_object) {
    if (_document == NULL)
      return;

    value_t key(name, _document->GetAllocator());

    value_t child_object(rapidjson::kObjectType);
    for (auto it = vec.begin(); it != vec.end(); ++it) {
      set_val(it->first.c_str(), it->second, child_object);
    }

    json_object.AddMember(key, child_object, _document->GetAllocator());
  }

  template <typename T>
  void set_val(std::string name, const T& val) {
    if (_document == NULL)
      return;

    set_val(name, val, *_document);
  }
  // set <end>

};  // class json

}  // namespace io
}  // namespace gunrock