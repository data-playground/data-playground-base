"""convert category enum to varchar and add categories table

Revision ID: b80965cfe3ab
Revises: m3rg3h34ds01
Create Date: 2026-03-25

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'b80965cfe3ac'
down_revision: Union[str, None] = 'm3rg3h34ds01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The original 8 categories to seed
SEED_CATEGORIES = [
    ("Housing",          "Rent, mortgage, utilities, home insurance"),
    ("Food & Dining",    "Restaurants, groceries, coffee, food delivery"),
    ("Transport",        "Uber, Lyft, gas, subway, parking, tolls"),
    ("Subscriptions",    "Netflix, Spotify, gym memberships, recurring SaaS"),
    ("Health",           "Doctor, pharmacy, insurance, dental, vision"),
    ("Entertainment",    "Movies, concerts, hobbies, streaming, games"),
    ("Savings Transfer", "Transfers to savings or investment accounts"),
    ("Income",           "Payroll, salary, freelance payments, refunds"),
    ("Other",            "Anything that does not fit another category"),
]


def upgrade() -> None:
    # 1. Create the categories table
    op.create_table(
        'categories',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('description', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    # 2. Seed the default categories
    op.bulk_insert(
        sa.table('categories',
            sa.column('name', sa.String),
            sa.column('description', sa.String),
            sa.column('is_active', sa.Boolean),
        ),
        [{"name": name, "description": desc, "is_active": True}
         for name, desc in SEED_CATEGORIES]
    )

    # 3. Alter transactions.category from ENUM → VARCHAR(100)
    #    MariaDB preserves existing string values during this conversion
    op.alter_column(
        'transactions',
        'category',
        existing_type=sa.Enum(
            'Housing', 'Food & Dining', 'Transport', 'Subscriptions',
            'Health', 'Entertainment', 'Savings Transfer', 'Income', 'Other',
            name='transactioncategory'
        ),
        type_=sa.String(100),
        existing_nullable=False,
        server_default='Other',
    )


def downgrade() -> None:
    # Revert VARCHAR back to ENUM
    op.alter_column(
        'transactions',
        'category',
        existing_type=sa.String(100),
        type_=sa.Enum(
            'Housing', 'Food & Dining', 'Transport', 'Subscriptions',
            'Health', 'Entertainment', 'Savings Transfer', 'Income', 'Other',
            name='transactioncategory'
        ),
        existing_nullable=False,
        server_default='Other',
    )
    op.drop_table('categories')
