// SQL AST + recursive-descent parser for the MySQL-flavoured subset gpu-db understands.
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "common.h"

namespace gdb {

struct Expr;
struct Stmt;
using ExprP = std::shared_ptr<Expr>;

struct Expr {
  enum Kind { COL, LIT, BIN, UNARY, FUNC, BETWEEN, INLIST, LIKE, CASE, STAR, INTERVAL, SUBQUERY } kind;
  std::string op;           // BIN/UNARY operator, FUNC name (upper case), INTERVAL unit
  std::string table, name;  // COL
  Value lit;                // LIT, INTERVAL amount in lit.i
  std::vector<ExprP> args;  // children. CASE: cond,val,cond,val,...[,else]
  bool neg = false;         // NOT BETWEEN / NOT IN / NOT LIKE
  bool distinct = false;    // COUNT(DISTINCT ..)
  int fnId = -1;            // scalar function id (FnId) for #CALL1 / #CALL2 nodes
  std::string castType;     // CAST(x AS <type>): upper-case target type
  int scale = -1;           // DECIMAL-typed expressions: digits after the point
  std::shared_ptr<Stmt> sub;  // SUBQUERY (scalar), INLIST (x IN (SELECT ..)), FUNC EXISTS
  std::shared_ptr<std::vector<double>> aux;  // "#MAP" code->value table, "#INSORTED" sorted values
  // Window functions: f(..) OVER (PARTITION BY .. ORDER BY .. [ROWS BETWEEN lo AND hi])
  bool window = false;
  std::vector<ExprP> partBy;
  std::vector<std::pair<ExprP, bool>> winOrder;  // (expression, descending)
  bool rowsFrame = false;
  int64_t frameLo = INT64_MIN, frameHi = 0;     // row offsets relative to the current row
  // Filled in by the binder.
  int bt = -1, bc = -1;     // COL: table index in FROM list, column index
  LType ty = LType::BIGINT;
  bool bound = false;
  std::string str() const;  // SQL-ish rendering (used for result column names)
};

ExprP mkCol(const std::string& table, const std::string& name);
ExprP mkLit(Value v);
ExprP mkBin(const std::string& op, ExprP a, ExprP b);

struct SelectItem {
  ExprP e;
  std::string alias;
};
struct TableRef {
  std::string name, alias;
  std::shared_ptr<Stmt> sub;  // derived table: FROM (SELECT ..) AS alias
  int join = 0;               // how this table joins the ones before it: 0 inner, 1 LEFT, 2 RIGHT, 3 FULL
  ExprP on;                   // ON condition of an outer join (inner join conditions go to joinConds)
  bool natural = false;       // NATURAL JOIN: equalities on the shared column names (added by the engine)
  std::vector<std::string> usingCols;  // JOIN .. USING (cols) / NATURAL: unqualified references to these are not ambiguous
  bool tvf = false;           // FROM fn(args): a table-valued function call (expanded by the engine)
  bool lateral = false;       // JOIN LATERAL (subquery): may reference earlier FROM entries
  std::vector<ExprP> tvfArgs;
};
struct OrderItem {
  ExprP e;
  bool desc = false;
};
struct ForeignKeyDef;
struct ColumnDef {
  std::string name;
  LType type;
  std::string typeText;  // e.g. "decimal(15,2)"
  bool notNull = false, autoInc = false, unique = false;
  ExprP def;             // DEFAULT expression (literal), null if none
  int vecDim = 0;        // VECTOR(n)
  int scale = 0;         // DECIMAL(p, s)
  int vram = 0;          // VRAM = AUTO (0) | PIN (1) | NEVER (2)
  std::string defText;   // DEFAULT (now()): the expression as written (information_schema)
  std::shared_ptr<ForeignKeyDef> ref;  // column-level REFERENCES t (c)
};
// FOREIGN KEY: kept as metadata (SHOW CREATE TABLE, information_schema) but not enforced.
struct ForeignKeyDef {
  std::string name;
  std::vector<std::string> cols, refCols;
  std::string refTable;
  std::string onDelete = "NO ACTION", onUpdate = "NO ACTION";
};
struct IndexDef {
  std::string name;
  std::vector<std::string> cols;
  bool unique = false;
  bool vector = false;   // VECTOR INDEX (IVF ANN index on a VECTOR column)
  std::string options;   // INDEX_OPTIONS '{"metric_type":"DOT_PRODUCT","nlist":..,"nprobe":..}' 
};

struct Stmt {
  enum Type {
    SELECT, INSERT, CREATE, DROP, DELETE, UPDATE, LOAD, SET, SHOW_TABLES, SHOW_DATABASES,
    SHOW_SEGMENTS, DESCRIBE, OPTIMIZE, EXPLAIN, USE, TXN, EMPTY,
    PROFILE, SHOW_PROFILE, SHOW_PLANCACHE, SHOW_PLAN, DROP_PLANCACHE,
    ALTER, TRUNCATE, RENAME, CREATE_DB, DROP_DB, CREATE_VIEW, DROP_VIEW, SHOW_MISC, PIPELINE, SNAPSHOT, USER, BACKUP
  } type = EMPTY;
  // SELECT
  bool distinct = false;
  std::vector<SelectItem> items;
  std::vector<TableRef> from;
  std::vector<ExprP> joinConds;
  ExprP where, having;
  std::vector<ExprP> groupBy;
  // ROLLUP / CUBE / GROUPING SETS: each set lists indexes into groupBy (empty: no grouping sets)
  std::vector<std::vector<int>> groupingSets;
  std::vector<OrderItem> orderBy;
  int64_t limit = -1, offset = 0;
  // Common
  std::string table;
  bool ifNotExists = false, ifExists = false;
  // CREATE
  std::vector<ColumnDef> cols;
  std::string sortKey;
  std::vector<std::string> pkCols;     // PRIMARY KEY columns
  std::vector<std::string> shardKey;   // SHARD KEY columns (empty + hasShardKey => keyless)
  bool hasShardKey = false;
  bool reference = false;              // CREATE REFERENCE TABLE: replicated, not sharded
  int vram = 0;                        // table option VRAM = AUTO | PIN | NEVER
  std::vector<ForeignKeyDef> fks;      // FOREIGN KEY constraints (not enforced)
  std::vector<std::string> fulltextCols;  // FULLTEXT [KEY] (cols): inverted indexes built eagerly
  std::vector<IndexDef> indexes;       // KEY / INDEX / UNIQUE KEY (secondary indexes)
  // INSERT
  std::vector<std::string> insertCols;
  std::vector<std::vector<ExprP>> rows;
  // UPDATE
  std::vector<std::pair<std::string, ExprP>> sets;
  // LOAD DATA
  std::string path;
  char delim = '\t';
  char escape = 0;  // LOAD DATA .. ESCAPED BY 'c' (0: no escape processing)
  int64_t skipLines = 0;
  std::string config, credentials, format;  // LOAD DATA S3/KAFKA .. CONFIG '{..}' CREDENTIALS '{..}' FORMAT JSON
  std::vector<std::string> jsonPaths;       // FORMAT JSON (col <- a::b, ..): source path per insertCols entry
  // EXPLAIN / PROFILE
  std::shared_ptr<Stmt> inner;
  // WITH name AS (..), ..: common table expressions; UNION chain; INSERT .. SELECT; CREATE .. AS SELECT
  std::vector<std::pair<std::string, std::shared_ptr<Stmt>>> ctes;
  std::shared_ptr<Stmt> unionNext;
  bool unionAll = false;
  bool unionOrder = false;       // ORDER BY / LIMIT apply to the whole UNION
  int setOp = 0;                 // how unionNext combines: 0 UNION, 1 INTERSECT, 2 EXCEPT
  std::shared_ptr<Stmt> select;  // INSERT .. SELECT / CREATE TABLE .. AS SELECT
  bool insertIgnore = false, replaceInto = false;
  std::vector<std::pair<std::string, ExprP>> onDup;  // ON DUPLICATE KEY UPDATE
  // ALTER TABLE actions
  struct Alter {
    enum Kind { ADD_COL, DROP_COL, RENAME_COL, MODIFY_COL, RENAME_TABLE, ADD_INDEX, DROP_INDEX, SET_VRAM, ADD_FK, DROP_FK, NOOP } kind = NOOP;
    ForeignKeyDef fk;
    int vram = 0;  // SET_VRAM: the hint; name: the column (empty: the table)
    ColumnDef col;
    std::string name, newName;
    IndexDef index;
  };
  std::vector<Alter> alters;
  std::vector<std::string> names;       // DROP TABLE a, b / RENAME TABLE a TO b, c TO d (pairs)
  std::string viewSql;                  // CREATE VIEW .. AS <sql>
  std::string outfile;                  // SELECT .. INTO OUTFILE 'path'
  char outDelim = '\t';
  bool temporary = false;               // CREATE TEMPORARY TABLE
  // SET name = value (only gpudb_* variables do anything); SHOW PLAN <id>
  std::string setName, setValue;
  std::string targetAlias;  // multi-table UPDATE / DELETE: the FROM entry that is modified
  int64_t planId = 0;
  std::vector<std::string> grantees;    // GRANT .. TO u, v / REVOKE .. FROM u
};
// The user the current statement runs as (CURRENT_USER()).
const std::string& sessionUser();
// Rewrites a call of a user-defined function while parsing (nullptr: not a UDF).
extern ExprP (*udfHook)(const ExprP&);
// Session user variables (SET @x = ..): NULL when unset.
Value userVar(const std::string& name);
std::string isolationLevel();
Value sysVarValue(const std::string& name);  // @@name (engine.cu): typed like MySQL; NULL if unknown
std::vector<std::pair<std::string, std::string>> systemVariables();  // the session's: READ-COMMITTED or REPEATABLE-READ (engine.cu)
std::string udfLiteral(const Value& v);  // a value as SQL literal text
Value callUdf(const std::string& name, const std::vector<std::string>& args);  // run a stored function
void setUserVar(const std::string& name, Value v);

Stmt parseSql(const std::string& sql);
std::string upper(std::string s);
// First keyword of a statement, upper case (cheap dispatch before running regexes over the text).
inline std::string firstWord(const std::string& s) {
  size_t i = 0;
  while (i < s.size() && (isspace((unsigned char)s[i]) || s[i] == '(')) ++i;
  size_t j = i;
  while (j < s.size() && isalpha((unsigned char)s[j]) && j - i < 16) ++j;
  std::string w = s.substr(i, j - i);
  for (auto& c : w) c = (char)toupper((unsigned char)c);
  return w;
}
bool likeMatch(const std::string& s, const std::string& pat);

}  // namespace gdb
