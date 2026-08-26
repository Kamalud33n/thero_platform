import re
import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker

# Load .env
load_dotenv()

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/rehab.db")

_debug_url = make_url(DATABASE_URL)
print(f"DATABASE_URL resolved to: "
      f"{_debug_url.drivername}://{_debug_url.username}:***@"
      f"{_debug_url.host}:{_debug_url.port}/{_debug_url.database}")


def _ensure_mysql_database_exists(db_url: str) -> None:

    import pymysql

    url = make_url(db_url)
    db_name = url.database
    if not db_name:
        return

    host = url.host or "localhost"
    port = url.port or 3306
    user = url.username
    print(f"Connecting to MySQL at {host}:{port} as '{user}' "
          f"to ensure database '{db_name}' exists...")

    conn = None
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=url.password or "",
            # no `database=` kwarg at all — connects to the server only
        )
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()
        print(f"MySQL database '{db_name}' ready (created if it didn't exist)")
    except Exception as exc:
        print(f"Could not auto-create MySQL database '{db_name}': {exc}")
        print("Check DATABASE_URL host/user/password in your .env file, "
              "and that the MySQL server is running.")
        raise
    finally:
        if conn is not None:
            conn.close()


# Engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_size=5,
        max_overflow=10,
    )
else:
    _ensure_mysql_database_exists(DATABASE_URL)
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,   # avoid MySQL "server has gone away" on idle conns
        pool_pre_ping=True,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _normalize_type_str(type_str: str) -> str:
    """
    Normalize a compiled/reflected column type string so equivalent types
    don't get falsely flagged as "out of sync" on every startup:

    - MySQL's information_schema reflects VARCHAR/TEXT columns with an
      explicit "COLLATE ..." (and sometimes "CHARACTER SET ...") suffix,
      but SQLAlchemy's own compiled model type never includes it — so a
      column that is actually fine would otherwise mismatch forever.
    - MySQL has no real BOOLEAN storage type; BOOL/BOOLEAN is just an
      alias for TINYINT(1). After ALTERing a column to BOOL, MySQL will
      always reflect it back as TINYINT(1) on the next inspection, so
      these two need to be treated as equivalent too.
    """
    s = re.sub(r"\s+COLLATE\s+\S+", "", type_str, flags=re.IGNORECASE)
    s = re.sub(r"\s+CHARACTER SET\s+\S+", "", s, flags=re.IGNORECASE)
    s = s.strip().upper()
    if s in ("BOOL", "BOOLEAN", "TINYINT(1)"):
        s = "TINYINT(1)"
    return s


def _auto_sync_mysql_columns() -> None:

    from sqlalchemy import inspect

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # brand new table — create_all() already built it correctly

            db_columns = {c["name"]: c for c in inspector.get_columns(table.name)}

            for column in table.columns:
                if column.primary_key:
                    continue  # never touch PK columns

                db_col = db_columns.get(column.name)
                if db_col is None:
                    # Column exists on the model but not yet in the live MySQL
                    # table (e.g. someone just added a new field to models.py).
                    # Add it automatically instead of requiring a manual ALTER.
                    try:
                        col_type_str = str(column.type.compile(dialect=engine.dialect))
                    except Exception as exc:
                        print(f"Could not compile type for new column "
                              f"'{table.name}.{column.name}': {exc}")
                        continue

                    nullability = "NULL" if column.nullable else "NOT NULL"
                    default_clause = ""
                    if column.default is not None and getattr(column.default, "is_scalar", False):
                        default_clause = f" DEFAULT {column.default.arg!r}"

                    print(f"Column '{table.name}.{column.name}' missing in MySQL. "
                          f"Auto-adding as {col_type_str}...")
                    try:
                        conn.execute(text(
                            f"ALTER TABLE `{table.name}` "
                            f"ADD COLUMN `{column.name}` {col_type_str} "
                            f"{nullability}{default_clause}"
                        ))
                        print(f"Column '{table.name}.{column.name}' added.")

                        if column.index or column.unique:
                            idx_name = f"ix_{table.name}_{column.name}"
                            unique_kw = "UNIQUE " if column.unique else ""
                            try:
                                conn.execute(text(
                                    f"CREATE {unique_kw}INDEX `{idx_name}` "
                                    f"ON `{table.name}` (`{column.name}`)"
                                ))
                                print(f"Index '{idx_name}' created.")
                            except Exception as exc:
                                print(f"Could not create index on "
                                      f"'{table.name}.{column.name}': {exc}")
                    except Exception as exc:
                        print(f"Could not auto-add column "
                              f"'{table.name}.{column.name}': {exc}")
                        print("You may need to add it manually with an ALTER TABLE statement.")
                    continue

                try:
                    model_type_str = str(column.type.compile(dialect=engine.dialect))
                    db_type_str = str(db_col["type"].compile(dialect=engine.dialect))
                except Exception:
                    continue  # if either type can't compile for this dialect, skip safely

                if _normalize_type_str(model_type_str) == _normalize_type_str(db_type_str):
                    continue  # already in sync (ignoring collation/charset noise + BOOL~TINYINT(1))

                nullability = "NULL" if column.nullable else "NOT NULL"
                print(f"Column '{table.name}.{column.name}' out of sync: "
                      f"{db_type_str} -> {model_type_str}. Auto-altering...")
                try:
                    conn.execute(text(
                        f"ALTER TABLE `{table.name}` "
                        f"MODIFY COLUMN `{column.name}` {model_type_str} {nullability}"
                    ))
                    print(f"Column '{table.name}.{column.name}' updated to {model_type_str}")
                except Exception as exc:
                    print(f"Could not auto-alter '{table.name}.{column.name}': {exc}")
                    print("You may need to update it manually with an ALTER TABLE statement.")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    if not DATABASE_URL.startswith("sqlite"):
        _auto_sync_mysql_columns()
    print(f"Database tables ready ({'MySQL' if not DATABASE_URL.startswith('sqlite') else 'SQLite'})")


@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()