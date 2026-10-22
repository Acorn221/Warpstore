// External data sources for LOAD DATA / CREATE PIPELINE: S3 (and S3-compatible stores), HTTP(S),
// Kafka. Everything ends up as a staged local file that the normal LOAD DATA INFILE path loads.
#pragma once
#include <cstdint>
#include <deque>
#include <map>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace gdb {

// Top-level string / number field of a flat JSON object ('' when absent).
std::string jsonField(const std::string& json, const std::string& key);

// S3: path is 'bucket/prefix' (a '*' / '?' in it filters keys like a glob). config: region,
// endpoint_url (S3-compatible stores); credentials: aws_access_key_id, aws_secret_access_key,
// aws_session_token.
struct S3Object {
  std::string key;
  int64_t size = 0;
};
std::vector<S3Object> s3List(const std::string& path, const std::string& config, const std::string& creds);
std::string s3Get(const std::string& bucket, const std::string& key, const std::string& config, const std::string& creds);
std::string s3Bucket(const std::string& path);
// Backups: a local file to / from an S3 object (streamed; multipart above 64 MB).
void s3PutFile(const std::string& bucket, const std::string& key, const std::string& file, const std::string& config, const std::string& creds);
void s3GetFile(const std::string& bucket, const std::string& key, const std::string& file, const std::string& config, const std::string& creds);
std::string httpGet(const std::string& url);
// POST body (JSON or raw bytes) and return the response body; throws SqlError on HTTP >= 300.
std::string httpPost(const std::string& url, const std::string& body, const std::string& contentType = "application/json");

// Kafka: path is 'host:port[,host:port]/topic'. next: partition -> next offset to read (filled with
// the earliest offsets for partitions not in it). Returns up to about maxBytes of message values.
struct KafkaMsg {
  int partition;
  int64_t offset;
  std::string value;
};
std::vector<KafkaMsg> kafkaPoll(const std::string& path, std::map<int, int64_t>& next, int32_t maxBytes);

// .gz / .zst / .lz4 objects are decompressed by name.
std::string decompressByName(const std::string& name, std::string data);

// Parquet file -> tab-separated rows of the named columns (by name; absent: \N). parquetColumns:
// the file's column names.
std::string parquetToTsv(std::string_view file, const std::vector<std::string>& cols);
// Flat Parquet columns decoded for a direct load: per row group, the wanted columns (by name) at
// their rows. kind 0: absent (all NULL), 1: integers (unit 0 number, 1 DATE days, 2 TIMESTAMP
// microseconds, 3 DECIMAL unscaled with `scale`, 4 BOOLEAN), 2: doubles, 3: strings (views into keep).
struct PqChunk {
  int kind = 0, unit = 0, scale = 0;
  std::vector<int64_t> i;
  std::vector<double> d;
  std::vector<std::string_view> s;
  std::vector<uint8_t> valid;  // per row
  std::deque<std::string> keep;  // (a deque: its strings never move, short ones included)
  // Strings from dictionary-encoded pages only (one dictionary): s[r] == dict[code[r]] for valid rows.
  std::vector<std::string_view> dict;
  std::vector<uint32_t> code;
};
// Calls fn(rows, columns) per row group (decoded in parallel). False, before any call, when a wanted
// column needs the text path (nested, TIME, decimals over 18 digits, unusual encodings).
bool parquetScan(std::string_view file, const std::vector<std::string>& wanted,
                 const std::function<void(int64_t, std::vector<PqChunk>&)>& fn);
std::vector<std::string> parquetColumns(std::string_view file);
// Avro object container file -> tab-separated rows of the named fields.
std::string avroToTsv(const std::string& file, const std::vector<std::string>& cols);
// ORC file -> tab-separated rows of the named columns.
std::string orcToTsv(const std::string& file, const std::vector<std::string>& cols);

// One JSON object per line -> tab-separated rows in the order of cols (missing keys: \N).
std::string jsonLinesToTsv(const std::string& data, const std::vector<std::string>& cols);

}  // namespace gdb
