const { PrismaClient, Prisma } = require("@prisma/client");
const p = new PrismaClient();
let ok = 0, fail = 0;
async function t(name, fn, chk = () => true) {
  try { const r = await fn(); if (chk(r)) { ok++; console.log("ok  ", name); } else { fail++; console.log("FAIL", name, JSON.stringify(r, (k, v) => typeof v === "bigint" ? v.toString() : v).slice(0, 300)); } }
  catch (e) { fail++; console.log("ERR ", name, String(e.message).split("\n").filter(Boolean).slice(-3).join(" | ").slice(0, 400)); }
}
(async () => {
  await t("deleteMany", async () => { await p.post.deleteMany(); await p.user.deleteMany(); });
  await t("create nested", () => p.user.create({ data: { email: "a@x", name: "Ann", balance: new Prisma.Decimal("10.5"), meta: { k: [1, 2] }, posts: { create: [{ title: "one" }, { title: "two", body: "b" }] } }, include: { posts: true } }), (r) => r.posts.length === 2 && r.meta.k[1] === 2);
  await t("createMany", () => p.user.createMany({ data: [{ email: "b@x", name: "Bob" }, { email: "c@x" }] }), (r) => r.count === 2);
  await t("findMany include + orderBy", () => p.user.findMany({ include: { posts: { orderBy: { id: "desc" } } }, orderBy: { id: "asc" } }), (r) => r.length === 3 && r[0].posts.length === 2 && r[0].posts[0].title === "two");
  await t("findUnique", () => p.user.findUnique({ where: { email: "a@x" } }), (r) => r && r.name === "Ann" && r.balance.toString() === "10.5" && r.active === true && r.createdAt instanceof Date);
  await t("where filters", () => p.user.findMany({ where: { OR: [{ name: { startsWith: "B" } }, { name: null }] } }), (r) => r.length === 2);
  await t("contains / in / not", () => p.post.findMany({ where: { title: { contains: "o" }, id: { notIn: [-1] }, NOT: { body: "zzz" } } }), (r) => r.length === 1);  // NOT (body = x) drops NULL bodies, as in MySQL
  await t("count / aggregate", () => p.post.aggregate({ _count: true, _max: { id: true } }), (r) => r._count === 2);
  await t("groupBy", () => p.post.groupBy({ by: ["authorId"], _count: { _all: true } }), (r) => r.length === 1 && r[0]._count._all === 2);
  await t("update", () => p.user.update({ where: { email: "b@x" }, data: { balance: { increment: 5 }, name: "Bobby" } }), (r) => r.balance.toString() === "5" && r.name === "Bobby");
  await t("upsert (update path)", () => p.user.upsert({ where: { email: "c@x" }, update: { name: "Cy" }, create: { email: "c@x" } }), (r) => r.name === "Cy");
  await t("upsert (create path)", () => p.user.upsert({ where: { email: "d@x" }, update: {}, create: { email: "d@x", name: "Di" } }), (r) => r.name === "Di");
  await t("unique violation P2002", async () => { try { await p.user.create({ data: { email: "a@x" } }); return "no error"; } catch (e) { return e.code; } }, (r) => r === "P2002");
  await t("interactive transaction rollback", async () => { try { await p.$transaction(async (tx) => { await tx.user.create({ data: { email: "t@x" } }); throw new Error("abort"); }); } catch {} return p.user.count({ where: { email: "t@x" } }); }, (r) => r === 0);
  await t("batch transaction", () => p.$transaction([p.user.count(), p.post.count()]), (r) => r[0] === 4 && r[1] === 2);
  await t("relation filter some", () => p.user.findMany({ where: { posts: { some: { title: "one" } } } }), (r) => r.length === 1);
  await t("select + _count relation", () => p.user.findMany({ select: { email: true, _count: { select: { posts: true } } }, orderBy: { email: "asc" } }), (r) => r[0]._count.posts === 2);
  await t("pagination skip/take", () => p.user.findMany({ skip: 1, take: 2, orderBy: { id: "asc" } }), (r) => r.length === 2);
  await t("raw query", () => p.$queryRaw`SELECT COUNT(*) AS n FROM User WHERE email LIKE ${"%x"}`, (r) => Number(r[0].n) === 4);
  await t("delete cascade manual", async () => { await p.post.deleteMany({ where: { author: { email: "a@x" } } }); return p.post.count(); }, (r) => r === 0);
  console.log(`${ok} passed, ${fail} failed`);
  await p.$disconnect();
  process.exit(fail ? 1 : 0);
})();
