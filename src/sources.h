// External data sources for LOAD DATA / CREATE PIPELINE: S3 (and S3-compatible stores), HTTP(S),
// Kafka. Everything ends up as a staged local file that the normal LOAD DATA INFILE path loads.
#pragma once
#include <cstdint>
#include <map>
#include <string>
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
std::string parquetToTsv(const std::string& file, const std::vector<std::string>& cols);
std::vector<std::string> parquetColumns(const std::string& file);
// Avro object container file -> tab-separated rows of the named fields.
std::string avroToTsv(const std::string& file, const std::vector<std::string>& cols);
// ORC file -> tab-separated rows of the named columns.
std::string orcToTsv(const std::string& file, const std::vector<std::string>& cols);

// One JSON object per line -> tab-separated rows in the order of cols (missing keys: \N).
std::string jsonLinesToTsv(const std::string& data, const std::vector<std::string>& cols);

}  // namespace gdb
