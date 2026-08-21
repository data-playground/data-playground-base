import datetime
import enum
from decimal import Decimal
from typing import Optional

from core.base_model import Base
from pydantic import BaseModel
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

"""
FINANCE MODULE — updated models for dynamic categories.
Replace the existing finance section in models.py with this.
"""

class AccountType(enum.Enum):
    CHECKING     = "Checking"
    CREDIT_CARD  = "Credit Card"
    SAVINGS      = "Savings"


# ── Dynamic category — no longer an Enum, now a DB-backed table ──

class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    transactions: Mapped[list["Transaction"]] = relationship(
        "Transaction", back_populates="category_obj"
    )


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    account_type: Mapped[AccountType] = mapped_column(
        Enum(AccountType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    last_four: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    transactions: Mapped[list["Transaction"]] = relationship(
        "Transaction", back_populates="account", lazy="selectin"
    )


class Transaction(Base):
    __tablename__ = "transactions"

    # BigInteger().with_variant(Integer, "sqlite"): SQLite's aiosqlite driver
    # only auto-generates rowids for a plain INTEGER primary key, not
    # BigInteger — the variant keeps BIGINT AUTO_INCREMENT on the real
    # MariaDB backend (unaffected) while making this table usable in fast
    # in-memory SQLite tests. See finance-migration-postmortem.md §5.2.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True, autoincrement=True,
    )
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)

    # category_id is the source of truth (FK -> categories.id), replacing
    # the old free-standing `category` string column. Nullable + SET NULL
    # so a category can be safely deleted in the future without corrupting
    # transaction rows (today the app has no delete-category endpoint, only
    # toggle-active, but the FK is defensive against that changing later).
    # Also nullable to tolerate legacy rows whose original category string
    # didn't match any live Category row at backfill time — see
    # domains/finance/migrations/0001_add_transaction_category_fk.py.
    category_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("categories.id", ondelete="SET NULL"), nullable=True, index=True
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    account: Mapped["Account"] = relationship("Account", back_populates="transactions")
    category_obj: Mapped[Optional["Category"]] = relationship(
        "Category", back_populates="transactions", lazy="selectin"
    )

    @property
    def category(self) -> str:
        """
        Backward-compatible string accessor. Every router/template that
        pre-dates this fix reads `t.category` expecting a plain string
        (e.g. `t.category.lower().replace(' ', '-')` for the CSS badge
        slug) — this property means none of that code needed to change.
        Falls back to "Other" for legacy rows that couldn't be matched to
        a real Category row during backfill, matching the pre-fix
        behavior where an unmatched category string also displayed as
        "Other" (see finance_upload.py's categorisation fallback).
        """
        return self.category_obj.name if self.category_obj else "Other"


# ── Pydantic schemas ──

class AccountCreate(BaseModel):
    name: str
    account_type: AccountType
    last_four: Optional[str] = None


class AccountResponse(BaseModel):
    id: int
    name: str
    account_type: AccountType
    last_four: Optional[str]

    class Config:
        from_attributes = True


class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None


class CategoryResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    is_active: bool

    class Config:
        from_attributes = True


class TransactionResponse(BaseModel):
    id: int
    account_id: int
    date: datetime.date
    description: str
    amount: Decimal
    category: str
    notes: Optional[str]

    class Config:
        from_attributes = True
