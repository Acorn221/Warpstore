import sys, datetime, decimal
from sqlalchemy import create_engine, String, Integer, ForeignKey, select, func, Numeric, DateTime, Boolean, JSON, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session, selectinload, joinedload
url = sys.argv[1]
eng = create_engine(url, echo=False)
with eng.connect() as c:
    c.execute(text("DROP DATABASE IF EXISTS sa")); c.execute(text("CREATE DATABASE sa")); c.commit()
eng = create_engine(url.rsplit("/", 1)[0] + "/sa")
class Base(DeclarativeBase): pass
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True)
    balance: Mapped[decimal.Decimal] = mapped_column(Numeric(10, 2), default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    meta: Mapped[dict] = mapped_column(JSON, nullable=True)
    created: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    posts: Mapped[list["Post"]] = relationship(back_populates="author")
class Post(Base):
    __tablename__ = "posts"
    id: Mapped[int] = mapped_column(primary_key=True)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(100))
    body: Mapped[str] = mapped_column(Text, nullable=True)
    author: Mapped[User] = relationship(back_populates="posts")
ok = fail = 0
def t(name, fn, chk=lambda r: True):
    global ok, fail
    try:
        r = fn()
        if chk(r): ok += 1; print("ok  ", name)
        else: fail += 1; print("FAIL", name, repr(r)[:200])
    except Exception as e:
        fail += 1; print("ERR ", name, str(e).split("\n")[0][:250])
t("create_all", lambda: Base.metadata.create_all(eng))
t("create_all again (checkfirst)", lambda: Base.metadata.create_all(eng))
with Session(eng) as s:
    t("add + commit", lambda: (s.add_all([User(name="Ann", email="a@x", balance=decimal.Decimal("10.50"), meta={"p": 1}, posts=[Post(title="hi", body="b"), Post(title="yo")]), User(name="Bob", email="b@x")]), s.commit()))
    t("query + relationship lazy", lambda: s.scalars(select(User).order_by(User.id)).first().posts, lambda r: len(r) == 2)
    t("selectinload", lambda: s.scalars(select(User).options(selectinload(User.posts)).order_by(User.id)).all(), lambda r: len(r) == 2 and len(r[0].posts) == 2)
    t("joinedload", lambda: s.scalars(select(Post).options(joinedload(Post.author)).order_by(Post.id)).all(), lambda r: r[0].author.name == "Ann")
    t("types roundtrip", lambda: s.scalars(select(User).where(User.email == "a@x")).one(), lambda u: u.balance == decimal.Decimal("10.50") and u.active is True and u.meta == {"p": 1} and isinstance(u.created, datetime.datetime))
    t("aggregate group by", lambda: s.execute(select(User.name, func.count(Post.id)).join(Post, isouter=True).group_by(User.name).order_by(User.name)).all(), lambda r: r == [("Ann", 2), ("Bob", 0)])
    def upd():
        u = s.scalars(select(User).where(User.name == "Bob")).one(); u.balance = decimal.Decimal("3.25"); s.commit()
        return s.scalar(select(User.balance).where(User.name == "Bob"))
    t("update via ORM", upd, lambda r: r == decimal.Decimal("3.25"))
    def rb():
        s.add(User(name="Tmp", email="t@x")); s.flush(); s.rollback()
        return s.scalar(select(func.count()).select_from(User).where(User.email == "t@x"))
    t("rollback", rb, lambda r: r == 0)
    def dup():
        try:
            s.add(User(name="Dup", email="a@x")); s.commit(); return "no error"
        except Exception as e:
            s.rollback(); return type(e).__name__
    t("unique violation -> IntegrityError", dup, lambda r: r == "IntegrityError")
    t("delete", lambda: (s.delete(s.scalars(select(Post).where(Post.title == "yo")).one()), s.commit()))
    t("raw text with params", lambda: s.execute(text("select count(*) from posts where title = :t"), {"t": "hi"}).scalar(), lambda r: r == 1)
t("reflect", lambda: __import__("sqlalchemy").MetaData().reflect(eng))
print(f"{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
