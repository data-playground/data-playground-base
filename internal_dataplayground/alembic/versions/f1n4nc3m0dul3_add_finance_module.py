"""add finance module tables

Revision ID: f1n4nc3m0dul3
Revises: a1b2c3d4e5f6
Create Date: 2026-03-14

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'f1n4nc3m0dul3'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'accounts',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('account_type', sa.Enum('Checking', 'Credit Card', 'Savings', name='accounttype'), nullable=False),
        sa.Column('last_four', sa.String(4), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        'transactions',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('account_id', sa.Integer(), sa.ForeignKey('accounts.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('date', sa.Date(), nullable=False, index=True),
        sa.Column('description', sa.String(500), nullable=False),
        sa.Column('amount', sa.Numeric(10, 2), nullable=False),  # negative=expense, positive=income
        sa.Column('category', sa.Enum(
            'Housing', 'Food & Dining', 'Transport', 'Subscriptions',
            'Health', 'Entertainment', 'Savings Transfer', 'Income', 'Other',
            name='transactioncategory'
        ), nullable=False, server_default='Other'),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_transactions_date_account', 'transactions', ['date', 'account_id'])


def downgrade() -> None:
    op.drop_index('ix_transactions_date_account', table_name='transactions')
    op.drop_table('transactions')
    op.drop_table('accounts')
    op.execute("DROP TYPE IF EXISTS transactioncategory")
    op.execute("DROP TYPE IF EXISTS accounttype")
