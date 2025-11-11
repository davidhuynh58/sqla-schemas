from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from enum import Enum
from typing import Annotated, Union, Optional, ClassVar, Any, Callable, Literal, Self, get_args
from sqlalchemy import (
    create_engine, MetaData,
    Column, 
    ForeignKey, 
    ForeignKeyConstraint, UniqueConstraint, CheckConstraint, Index,
    select, and_, or_,
)
from sqlalchemy.types import (
    Boolean, Integer, BigInteger, Float, String, PickleType, NullType, TIMESTAMP
)
from sqlalchemy.types import Enum as SQLAlchemyEnum
from sqlalchemy.sql import base as sqlbase
from sqlalchemy.dialects.postgresql import (JSONB, ARRAY)
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import sessionmaker, Session, declarative_base, DeclarativeBase
from sqlalchemy.exc import SQLAlchemyError
from operator import lt, le, eq, ne, ge, gt
import contextlib
import inspect
import json
from datetime import datetime

mutable_JSONB = MutableDict.as_mutable(JSONB)

meta = MetaData()
Base = declarative_base()

type_annotation_mapping_postgres = {
    datetime: TIMESTAMP(timezone=True)
}

class SqlaSchema(BaseModel):
    @classmethod
    def _convert_py_type(cls, py_type: type | Any, target: Literal["SQLAlchemy"] = "SQLAlchemy"):
        assert target in ["SQLAlchemy"], f"{target=} not supported"
        
        # check for underlying types, i.e. type(py_type) = types.GenericAlias
        # e.g. Union, Optional, list
        underlying_types = get_args(py_type)

        # handle types.GenericAlias
        if underlying_types:
            # handle list
            if len(underlying_types) == 1:
                if target == "SQLAlchemy":
                    # postgres supports ARRAY type
                    if 'postgres' in cls._db_url.lower():
                        return {'type_': ARRAY(cls._convert_py_type(underlying_types[0])['type_']), 'nullable': False}
                    
                    # otherwise, default to PickleType
                    else:
                        return {'type_': PickleType, 'nullable': False}

            # handle Union
            else:
                # determine if Optional
                isOptional = any([issubclass(underlying_type, type(None)) for underlying_type in underlying_types])

                # get non-None type
                underlying_types = [underlying_type for underlying_type in underlying_types if not issubclass(underlying_type, type(None))]

                if target == "SQLAlchemy":
                    return {'type_': cls._convert_py_type(underlying_types[0])['type_'], 'nullable': isOptional}

        # handle individual types
        else:
            if issubclass(py_type, bool):
                if target == "SQLAlchemy":
                    return {'type_': Boolean, 'nullable': False}
                
            elif issubclass(py_type, int):
                if target == "SQLAlchemy":
                    return {'type_': Integer, 'nullable': False}
                
            elif issubclass(py_type, float):
                if target == "SQLAlchemy":
                    return {'type_': Float, 'nullable': False}
                
            elif issubclass(py_type, str):
                if target == "SQLAlchemy":
                    return {'type_': String, 'nullable': False}
                
            elif issubclass(py_type, Enum):
                if target == "SQLAlchemy":
                    return {'type_': SQLAlchemyEnum(py_type), 'nullable': False}
                
            elif issubclass(py_type, BaseModel):
                if target == "SQLAlchemy":
                    # postgres supports ARRAY type
                    if 'postgres' in cls._db_url.lower():
                        return {'type_': mutable_JSONB, 'nullable': False}
                    
                    # otherwise, default to PickleType
                    else:
                        return {'type_': PickleType, 'nullable': False}
                    
            else:
                print(f"{py_type} type not supported")
                raise
        
    # class variables for SQLAlchemy setup
    _db_url: ClassVar[str] = "sqlite:///default.db"
    _base: ClassVar[Optional[DeclarativeBase]] = None
    _engine: ClassVar[Optional[Any]] = None
    _table: ClassVar[Optional[Any]] = None
    
    # private attribute for table arguments
    _table_args: list[
        Union[ForeignKeyConstraint, UniqueConstraint, CheckConstraint, Index]
    ] = []

    # method to convert results to dicts
    @classmethod
    def _results_to_dict(cls, results):
        return [
            {
                column.name: getattr(row, column.name)
                for column in cls._table.__table__.columns
            }
            for row in results
        ]
    
    @classmethod
    def _try_commit(cls, session: Session):
        try:
            # attempt to commit
            session.commit()
        except SQLAlchemyError as e:
            # Handle the general SQLAlchemy error
            session.rollback() # Important for maintaining session integrity
            print(f"An SQLAlchemy error occurred: {e}")
            raise

    # app-level method to construct base schema
    @classmethod
    def construct_base_schema(cls, db_url=None) -> Self:
        # set database url
        if db_url:
            cls._db_url = db_url

        # construct declarative base with appropriate type mapping
        if 'postgres' in cls._db_url.lower():
            cls._base = declarative_base(type_annotation_map=None)
        else:
            cls._base = declarative_base(type_annotation_map=None)

        return cls

    # app-level method to init db 
    @classmethod
    def sqla_init_db(cls, drop_tables_first=True):
        # create engine
        cls._engine = create_engine(cls._db_url)

        # drop all tables
        if drop_tables_first:
            Base.metadata.drop_all(cls._engine)

        # create all tables
        Base.metadata.create_all(cls._engine, checkfirst=True)
        
        # # truncate all tables
        # if truncate_tables:
        #     with contextlib.closing(cls._engine.connect()) as conn:
        #         trans = conn.begin()
        #         meta.reflect(bind=cls._engine)
        #         for table in reversed(meta.sorted_tables):
        #             conn.execute(table.delete())
        #         trans.commit()

    # schema-level method to create SQLAlchemy table from schema fields
    @classmethod
    def sqla_create_table_class(cls):
        # function for __str__ method of table class
        def str_method(self):
            output = {}
            for column in self.__table__.columns:
                output[column.name] = getattr(self, column.name)
            return json.dumps(output)
        
        # default primary key
        default_sql_pid = 'sql_pid'
        
        # init table definition dict with default primary key
        table_def = {
            '__tablename__': cls.__name__.lower(),
            '__str__': str_method,
            default_sql_pid: Column(Integer, primary_key=True),
        }

        # get Column for each field
        for field_name, field_info in cls.__pydantic_fields__.items():
            # generate default args based on schema type annotation
            default_args = cls._convert_py_type(field_info.annotation, target="SQLAlchemy")
            if not field_info.is_required():
                default_args['default'] = field_info.default

            # construct default Column
            default_column = Column(**default_args)
            
            # get Column if provided in schema
            field_column = [metadata for metadata in field_info.metadata if isinstance(metadata, Column)]
            if field_column:
                field_column = field_column[0]

                # if column type is NullType, assume that default Column values are desired
                # remove 'type' and 'nullable' fields from input args
                if isinstance(field_column.type, NullType):
                    field_column.type = default_column.type
                    field_column.nullable = default_column.nullable

                # if column is primary key, remove default primary key from table
                if field_column.primary_key:
                    table_def.pop(default_sql_pid)
                
                # if no default given in Column, use default Column
                if isinstance(field_column.default, type(None)):
                    field_column.default = default_column.default
                    pass

            # otherwise, use default Column for field
            else:
                field_column = default_column

            # update table definition with field and Column
            table_def[field_name] = field_column
        
        # create table class as attribute of schema class
        cls._table = type(cls.__name__, (Base,), table_def)
        pass

    # instance-level method to add instance as row to table
    def sqla_add(self):
        # construct session
        Session = sessionmaker(bind=self._engine)

        with Session() as session:
            # 4. Create an instance of your mapped class
            new_row = self._table(**self.model_dump())

            # 5. Add the instance to the session
            session.add(new_row)

            # 6. Commit the session to persist the changes to the database
            self._try_commit(session)

            print(f"Row added to {self._table.__tablename__} table: {self.model_dump()}")

    # schema-level method to bulk add instances to a table
    @classmethod
    def sqla_add_all(cls, objs: Self) -> list[Self]: 
        # construct session
        Session = sessionmaker(bind=cls._engine)

        with Session() as session:
            # map objects to table instances
            mapped_objs = [obj._table(**obj.model_dump()) for obj in objs]

            # add all objects to tables
            session.add_all(mapped_objs)

            # commit session to db
            cls._try_commit(session)

    # schema-level method to select SQLAlchemy table
    @classmethod
    def sqla_select_table(cls, where_conditions=None, bool_op=and_) -> list[Self]: 
        # construct where statement from conditions and boolean operation
        # TODO: add nested attribute option
        def _construct_where_condition(table, bool_op, conditions: list[tuple] | list = None):
            # if no conditions, return true for full select
            if not conditions:
                return True
            
            # if single condition given, wrap into list
            if not isinstance(conditions[0], (list, tuple)):
                conditions = [conditions]

            # construct where statement    
            return bool_op(*[condition[1](getattr(table, condition[0]), condition[2]) for condition in conditions])
            
        # construct select statement on table
        stmt = select(cls._table).where(
            _construct_where_condition(
                table=cls._table, bool_op=bool_op, conditions=where_conditions
            )
        )
        
        # construct session
        Session = sessionmaker(bind=cls._engine)

        with Session() as session:
            results = session.execute(stmt)

            formatted_results = [
                cls(
                    **{
                        column.name: getattr(obj, column.name)
                        for column in cls._table.__table__.columns
                    }
                ) for obj in results.scalars()
            ]
        
        print(f"{len(formatted_results)} result{'s' if len(formatted_results) > 1 else ''} retrieved from {cls._table.__tablename__} table")
        return formatted_results

if __name__ == "__main__":
    # define application schema class with url
    class AppSchema(SqlaSchema):
        _db_url = "sqlite:///test.db"

    # define application class, User
    class User(AppSchema):
        id: Annotated[str, Field(description="user id"), Column(String)]
    User.sqla_create_table_class()
    
    # define pydantic model to test nested models
    class NestedModel(BaseModel):
        id: str
        value: float

    # define enum to test
    class Color(str, Enum):
        RED = 'RED'
        GREEN = 'GREEN'
        BLUE = 'BLUE'

    # define application class, Resource
    class Resource(AppSchema):
        id: Annotated[str, Field(description="resource id"), Column(unique=False)]
        str_list: Annotated[list[str], Field(description="test list")]
        nested_model: Annotated[NestedModel, Field(description="test nested model")]
        enum_field: Color
        optional_field: Optional[float] = Field(default=None)
        nonrequired_field: Annotated[float, Field(default=0.)]
    Resource.sqla_create_table_class()
    
    AppSchema.sqla_init_db()

    user_01 = User(id="test_user_01")
    user_02 = User(id="test_user_02")
    user_01.sqla_add()

    resource_01 = Resource(
        id="test_resource_01", str_list=['1', '2'], 
        nested_model={'id': 'nest_01', 'value': 2.},
        enum_field=Color.RED,
        nonrequired_field=1
    )
    resource_02 = Resource(
        id="test_resource_02", str_list=['1', '2'], 
        nested_model={'id': 'nest_02', 'value': 2.},
        enum_field=Color.BLUE,
        nonrequired_field=1
    )

    # test bulk add
    AppSchema.sqla_add_all([user_02, resource_01, resource_02])

    # test select table
    users = User.sqla_select_table()
    print(users)
    
    resources = Resource.sqla_select_table()
    print(resources)
    
    # test select table with conditions
    resources = Resource.sqla_select_table(
        where_conditions=[
            ['id', eq, 'test_resource_01'],
            ['enum_field', eq, Color.BLUE],
        ],
        bool_op=or_
    )
    print(resources)