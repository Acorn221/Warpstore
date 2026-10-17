import { mysqlTable, int, bigint, varchar, text, timestamp, boolean, json, decimal, mysqlEnum, double, date, datetime, index, uniqueIndex, primaryKey } from "drizzle-orm/mysql-core";
import { relations, sql } from "drizzle-orm";

export const users = mysqlTable("users", {
  id: int("id").autoincrement().primaryKey(),
  email: varchar("email", { length: 255 }).notNull().unique(),
  name: varchar("name", { length: 100 }).notNull(),
  role: mysqlEnum("role", ["admin", "member"]).notNull().default("member"),
  active: boolean("active").notNull().default(true),
  balance: decimal("balance", { precision: 12, scale: 2 }).notNull().default("0.00"),
  meta: json("meta").$type<{ plan?: string; tags?: string[] }>(),
  createdAt: timestamp("created_at").notNull().defaultNow(),
  updatedAt: datetime("updated_at"),
}, (t) => [index("users_name_idx").on(t.name)]);

export const posts = mysqlTable("posts", {
  id: bigint("id", { mode: "number" }).autoincrement().primaryKey(),
  authorId: int("author_id").notNull().references(() => users.id),
  title: varchar("title", { length: 200 }).notNull(),
  body: text("body"),
  score: double("score").default(0),
  published: date("published", { mode: "string" }),
}, (t) => [index("posts_author_idx").on(t.authorId)]);

export const tags = mysqlTable("post_tags", {
  postId: bigint("post_id", { mode: "number" }).notNull().references(() => posts.id, { onDelete: "cascade" }),
  tag: varchar("tag", { length: 50 }).notNull(),
}, (t) => [primaryKey({ columns: [t.postId, t.tag] })]);

export const usersRelations = relations(users, ({ many }) => ({ posts: many(posts) }));
export const postsRelations = relations(posts, ({ one, many }) => ({
  author: one(users, { fields: [posts.authorId], references: [users.id] }),
  tags: many(tags),
}));
export const tagsRelations = relations(tags, ({ one }) => ({ post: one(posts, { fields: [tags.postId], references: [posts.id] }) }));
