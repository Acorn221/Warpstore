import { drizzle } from "drizzle-orm/mysql2";
import mysql from "mysql2/promise";
import { eq, and, or, gt, like, desc, asc, sql, count, sum, avg, inArray, isNull, between } from "drizzle-orm";
import * as schema from "./schema";
const { users, posts, tags } = schema;

let pass = 0, fail = 0;
async function t(name: string, fn: () => Promise<any>, check?: (r: any) => boolean) {
  try {
    const r = await fn();
    if (check && !check(r)) { fail++; console.log("FAIL", name, JSON.stringify(r)?.slice(0, 300)); }
    else { pass++; console.log("ok  ", name); }
  } catch (e: any) { fail++; console.log("ERR ", name, (e.sqlMessage || e.message || String(e)).slice(0, 300), e.sql ? "\n     SQL: " + e.sql.slice(0, 400) : ""); }
}

async function main() {
  const pool = mysql.createPool({ host: "127.0.0.1", port: Number(process.env.GPUDB_PORT || 3307), user: "root", database: "app", connectionLimit: 4 });
  const db = drizzle(pool, { schema, mode: "default" });
  await t("delete all", async () => { await db.delete(tags); await db.delete(posts); await db.delete(users); });
  await t("insert users", () => db.insert(users).values([
    { email: "a@x.io", name: "Ann", role: "admin", balance: "10.50", meta: { plan: "pro", tags: ["x"] } },
    { email: "b@x.io", name: "Bob" }, { email: "c@x.io", name: "Cy", active: false }]), (r) => r[0].affectedRows === 3);
  const ids = await db.select({ id: users.id, email: users.email }).from(users).orderBy(users.email);
  await t("insertId / $returningId", async () => db.insert(users).values({ email: "d@x.io", name: "Di" }).$returningId(), (r) => r.length === 1 && r[0].id > 0);
  await t("select all", () => db.select().from(users).orderBy(users.id), (r) => r.length === 4 && r[0].active === true && r[0].meta?.plan === "pro" && r[0].createdAt instanceof Date && r[0].balance === "10.50");
  await t("where eq + and/or", () => db.select().from(users).where(or(eq(users.role, "admin"), and(eq(users.active, false), like(users.name, "C%")))), (r) => r.length === 2);
  const a = ids[0].id, b = ids[1].id;
  await t("insert posts", () => db.insert(posts).values([
    { authorId: a, title: "Hello", body: "first", score: 4.5, published: "2024-05-01" },
    { authorId: a, title: "Again", score: 1.25 }, { authorId: b, title: "Bob post", score: 3 }]));
  const pid = (await db.select({ id: posts.id }).from(posts).orderBy(posts.id))[0].id;
  await t("insert tags (composite pk)", () => db.insert(tags).values([{ postId: pid, tag: "intro" }, { postId: pid, tag: "news" }]));
  await t("duplicate composite pk rejected", async () => { try { await db.insert(tags).values({ postId: pid, tag: "intro" }); return "no error"; } catch (e: any) { return e.cause?.code ?? e.code; } }, (r) => r === "ER_DUP_ENTRY");
  await t("onDuplicateKeyUpdate", () => db.insert(users).values({ email: "a@x.io", name: "Ann2" }).onDuplicateKeyUpdate({ set: { name: sql`values(${users.name})` } }));
  await t("inner join", () => db.select({ u: users.name, p: posts.title }).from(posts).innerJoin(users, eq(posts.authorId, users.id)).orderBy(posts.id), (r) => r.length === 3 && r[0].u === "Ann2");
  await t("left join nested objects", () => db.select().from(users).leftJoin(posts, eq(posts.authorId, users.id)).orderBy(users.id, posts.id), (r) => r.length === 5 && r[0].users && "posts" in r[0]);
  await t("aggregates + groupBy + having", () => db.select({ author: posts.authorId, n: count(), s: sum(posts.score), a: avg(posts.score) }).from(posts).groupBy(posts.authorId).having(gt(count(), 1)), (r) => r.length === 1 && r[0].n === 2);
  await t("count()", () => db.$count(users), (r) => r === 4);
  await t("count with filter", () => db.$count(users, eq(users.active, true)), (r) => r === 3);
  await t("inArray / isNull / between", () => db.select().from(posts).where(and(inArray(posts.authorId, [a, b]), isNull(posts.body), between(posts.score, 1, 5))), (r) => r.length === 2);
  await t("limit/offset/order desc", () => db.select().from(posts).orderBy(desc(posts.score)).limit(2).offset(1), (r) => r.length === 2 && r[0].score === 3);
  await t("update", () => db.update(users).set({ balance: sql`${users.balance} + 5`, updatedAt: new Date("2024-06-01T10:00:00Z") }).where(eq(users.id, a)), (r) => r[0].affectedRows === 1);
  await t("read back update", () => db.select({ b: users.balance, u: users.updatedAt }).from(users).where(eq(users.id, a)), (r) => r[0].b === "15.50" && r[0].u instanceof Date);
  await t("json column filter (->>)", () => db.select().from(users).where(sql`${users.meta}->>'$.plan' = 'pro'`), (r) => r.length === 1);
  await t("relational findMany with posts", () => db.query.users.findMany({ with: { posts: true }, orderBy: (u, { asc }) => [asc(u.id)] }), (r) => r.length === 4 && r[0].posts.length === 2);
  await t("relational findFirst nested", () => db.query.posts.findFirst({ where: eq(posts.id, pid), with: { author: true, tags: { orderBy: (t, { asc }) => [asc(t.tag)] } } }), (r) => r.author.name === "Ann2" && r.tags.length === 2 && r.tags[0].tag === "intro");
  await t("relational columns + extras", () => db.query.users.findMany({ columns: { id: true, name: true }, extras: { lower: sql<string>`lower(${users.name})`.as("lower") }, limit: 2 }), (r) => r.length === 2 && typeof r[0].lower === "string");
  await t("transaction commit", () => db.transaction(async (tx) => { await tx.insert(users).values({ email: "t@x.io", name: "Tx" }); }));
  await t("transaction rollback", async () => { try { await db.transaction(async (tx) => { await tx.insert(users).values({ email: "r@x.io", name: "Rb" }); tx.rollback(); }); } catch {} return db.$count(users, eq(users.email, "r@x.io")); }, (r) => r === 0);
  await t("nested transaction (savepoint)", () => db.transaction(async (tx) => { await tx.insert(users).values({ email: "s1@x.io", name: "S1" }); try { await tx.transaction(async (t2) => { await t2.insert(users).values({ email: "s2@x.io", name: "S2" }); t2.rollback(); }); } catch {} }));
  await t("subquery in where", () => db.select().from(users).where(inArray(users.id, db.select({ id: posts.authorId }).from(posts))), (r) => r.length === 2);
  await t("delete with where", () => db.delete(users).where(eq(users.email, "t@x.io")), (r) => r[0].affectedRows === 1);
  await t("raw execute", () => db.execute(sql`select 1 + ${2} as x`), (r) => r[0][0].x === 3);
  await t("prepared statement", async () => { const p = db.select().from(users).where(eq(users.id, sql.placeholder("id"))).prepare(); return p.execute({ id: a }); }, (r) => r.length === 1);
  console.log(`\n${pass} passed, ${fail} failed`);
  await pool.end();
  process.exit(fail ? 1 : 0);
}
main();
