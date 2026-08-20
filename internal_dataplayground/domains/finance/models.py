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

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    # category is now a plain string referencing categories.name
    category: Mapped[str] = mapped_column(String(100), nullable=False, default="Other")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    account: Mapped["Account"] = relationship("Account", back_populates="transactions")


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
