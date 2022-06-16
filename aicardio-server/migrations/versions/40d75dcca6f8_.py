"""empty message

Revision ID: 40d75dcca6f8
Revises: 4d67e2358cc1
Create Date: 2019-09-12 23:43:26.904979

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '40d75dcca6f8'
down_revision = '4d67e2358cc1'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('fullname', table_name='users')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index('fullname', 'users', ['fullname'], unique=True)
    # ### end Alembic commands ###