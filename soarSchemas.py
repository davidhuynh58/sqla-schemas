from pydantic import BaseModel, Field
from typing import Annotated, Union, Optional, ClassVar, Any
from sqlalchemy import (
    Column, 
    Integer, Float, String,
    ForeignKey, 
    ForeignKeyConstraint, UniqueConstraint, CheckConstraint, Index,
    select,
)

from sqla_schemas import SqlaSchema

if __name__ == "__main__":
    class SoarSchema(SqlaSchema):
        _url = "sqlite:///test.db"

    class Task(SoarSchema):
        id: Annotated[str, Field(description="task id"), Column(String, primary_key=True)]
    Task.sqla_create_table_class()
    
    class Opportunity(SoarSchema):
        id: Annotated[str, Field(description="opportunity id"), Column(String, primary_key=True)]
    Opportunity.sqla_create_table_class()
    
    SoarSchema.sqla_init_db()

    task_01 = Task(id="test_task")
    task_01.sqla_add()

    opportunity_01 = Opportunity(id="test_opportunity")
    opportunity_01.sqla_add()

    Task.sqla_select_table()
    task_01.sqla_select_table()

    Opportunity.sqla_select_table()
    opportunity_01.sqla_select_table()