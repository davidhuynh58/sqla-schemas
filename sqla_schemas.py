from pydantic import BaseModel, Field
from enum import Enum
from typing import Annotated, Union, Optional, ClassVar, Any, Literal, Self, get_origin, get_args
from sqlalchemy import (
    create_engine, Engine, MetaData,
    Column, 
    ForeignKey, 
    ForeignKeyConstraint, UniqueConstraint, CheckConstraint, Index,
    select, delete, and_, or_,
)
from sqlalchemy.types import (
    Boolean, Integer, BigInteger, Float, String, PickleType, NullType, TIMESTAMP, JSON,
)
from sqlalchemy.types import Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import (JSONB, ARRAY)
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import sessionmaker, Session, declarative_base, DeclarativeBase
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import BinaryExpression
import contextlib
import json
from datetime import datetime

mutable_JSONB = MutableDict.as_mutable(JSONB)

type_annotation_mapping_postgres = {
    datetime: TIMESTAMP(timezone=True),
    BaseModel: mutable_JSONB,
    list[Any]: mutable_JSONB,
    dict[str, Any]: mutable_JSONB,
}

type_annotation_mapping_nonpostgres = {
    datetime: TIMESTAMP(timezone=True),
    BaseModel: PickleType,
    list[Any]: PickleType,
    dict[str, Any]: PickleType,
}

class SqlaSchema(BaseModel):
    # class variables for SQLAlchemy setup
    db_url: ClassVar[str] = "sqlite:///default.db"
    sqla_table: ClassVar[Optional[DeclarativeBase]] = None
    _base: ClassVar[Optional[DeclarativeBase]] = None
    _meta: ClassVar[Optional[MetaData]] = None
    _engine: ClassVar[Optional[Engine]] = None
    __table_args__: ClassVar[Optional[Union[tuple, dict]]] = None

    @classmethod
    def _interpret_py_annotation(cls, py_anno: type | Any, target: Literal["SQLAlchemy"] = "SQLAlchemy"):
        assert target in ["SQLAlchemy"], f"{target=} not supported"
        
        # check for underlying types, i.e. type(py_type) = types.GenericAlias
        # e.g. Union, Optional, list
        underlying_types = get_args(py_anno)
        type_origin = get_origin(py_anno)

        # handle types.GenericAlias
        if underlying_types:
            # handle Union since cannot be evaluated by issubclass()
            if type_origin is Union:
                # determine if Optional
                isOptional = any([issubclass(underlying_type, type(None)) for underlying_type in underlying_types])

                # get non-None type
                underlying_types = [underlying_type for underlying_type in underlying_types if not issubclass(underlying_type, type(None))]

                if target == "SQLAlchemy":
                    return {'type_': cls._interpret_py_annotation(underlying_types[0])['type_'], 'nullable': isOptional}
            # handle list
            elif issubclass(type_origin, list):
                if target == "SQLAlchemy":
                    # postgres supports ARRAY type
                    if 'postgres' in cls.db_url.lower():
                        return {'type_': ARRAY(cls._interpret_py_annotation(underlying_types[0])['type_']), 'nullable': False}
                    
                    # otherwise, default to JSON
                    else:
                        return {'type_': JSON, 'nullable': False}
            else:
                print(f"{py_anno} not supported, only Union/Optional and list")
                pass

        # handle individual types
        else:
            if issubclass(py_anno, bool):
                if target == "SQLAlchemy":
                    return {'type_': Boolean, 'nullable': False}
                
            elif issubclass(py_anno, int):
                if target == "SQLAlchemy":
                    return {'type_': Integer, 'nullable': False}
                
            elif issubclass(py_anno, float):
                if target == "SQLAlchemy":
                    return {'type_': Float, 'nullable': False}
                
            elif issubclass(py_anno, str):
                if target == "SQLAlchemy":
                    return {'type_': String, 'nullable': False}
                
            elif issubclass(py_anno, Enum):
                if target == "SQLAlchemy":
                    return {'type_': SQLAlchemyEnum(py_anno), 'nullable': False}
                
            elif issubclass(py_anno, datetime):
                if target == "SQLAlchemy":
                    return {'type_': TIMESTAMP(timezone=True), 'nullable': False}
                
            elif issubclass(py_anno, BaseModel):
                if target == "SQLAlchemy":
                    # postgres supports mutable JSONB type
                    if 'postgres' in cls.db_url.lower():
                        return {'type_': mutable_JSONB, 'nullable': False}
                    
                    # otherwise, default to JSON
                    else:
                        return {'type_': JSON, 'nullable': False}
                    
            else:
                print(f"{py_anno} type not supported")
                raise
        
    # method to convert results to dicts
    @classmethod
    def _results_to_dict(cls, results):
        return [
            {
                column.name: getattr(row, column.name)
                for column in cls.sqla_table.__table__.columns
            }
            for row in results
        ]
    
    # app-level method to construct base schema
    @classmethod
    def construct_base_schema(cls, db_url=None) -> Self:
        # set database url
        if db_url:
            cls.db_url = db_url

        cls._base = declarative_base()
        cls._meta = MetaData()
        # type_annotation_map does not work so well with complex types e.g. nested classes or lists
        # # construct declarative base with appropriate type mapping
        # if 'postgres' in cls._db_url.lower():
        #     cls._base = declarative_base(type_annotation_map=type_annotation_mapping_postgres)
        # else:
        #     cls._base = declarative_base(type_annotation_map=type_annotation_mapping_nonpostgres)

        return cls

    # app-level method to init db 
    @classmethod
    def sqla_init_db(cls, drop_tables_first=True):
        # create engine
        print("Creating database engine")
        cls._engine = create_engine(cls.db_url)

        # drop all tables
        if drop_tables_first:
            print("Dropping any existing tables")
            cls._base.metadata.drop_all(cls._engine)

        # create all tables
        print("Creating all tables")
        cls._base.metadata.create_all(cls._engine, checkfirst=True)
        cls._meta.reflect(bind=cls._engine)
        print(f"{len(cls._meta.tables)} table(s) created: {', '.join([table.name for table in cls._meta.sorted_tables])}")
        
    # app/schema-level method to truncate tables
    @classmethod
    def sqla_truncate_table(cls):
        # if schema-level call, get table name
        if cls.sqla_table is not None:
            specific_table_to_delete = cls.sqla_table.__table__.name
        else:
            specific_table_to_delete = False

        # truncate table(s)
        with contextlib.closing(cls._engine.connect()) as conn:
            trans = conn.begin()
            cls._meta.reflect(bind=cls._engine)
            for table in reversed(cls._meta.sorted_tables):
                if not specific_table_to_delete or table.name == specific_table_to_delete:
                    print(f"Truncating '{table.name}' table")
                    conn.execute(table.delete())
            trans.commit()

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
        
        # format __table_args__ to be dict or tuple
        if cls.__table_args__ is not None and not isinstance(cls.__table_args__, (dict, tuple)):
            # convert to tuple
            cls.__table_args__ = tuple(cls.__table_args__)

        # init table definition dict with default primary key
        table_def = {
            '__tablename__': cls.__name__.lower(),
            '__annotations__': {},
            '__str__': str_method,
            '__table_args__': cls.__table_args__,
            default_sql_pid: Column(Integer, primary_key=True),
            # default_sql_pid: mapped_column(Integer, primary_key=True),
        }

        # get Column for each field
        for field_name, field_info in cls.__pydantic_fields__.items():
            # generate default args based on schema type annotation
            default_args = cls._interpret_py_annotation(field_info.annotation, target="SQLAlchemy")
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

        # # get Column for each field
        # for field_name, field_info in cls.__pydantic_fields__.items():
        #     # set table annotation as mapped annotation from schema
        #     table_def['__annotations__'][field_name] = Mapped[field_info.annotation]

        #     # if a mapped column or column was given, added to table definition
        #     field_column = [metadata for metadata in field_info.metadata if isinstance(metadata, (MappedColumn, Column))]
        #     if field_column:
        #         table_def[field_name] = field_column[0]
        
        # create table class as attribute of schema class
        cls.sqla_table = type(cls.__name__, (cls._base,), table_def)
        pass

    # schema-level method to execute statement
    @classmethod
    def sqla_execute(cls, stmt) -> list[Self]: 
        # construct session
        Session = sessionmaker(bind=cls._engine)

        with Session() as session:
            results = session.execute(stmt).all()
        
        return results

    # schema-level method to select entire table with whereclause
    @classmethod
    def sqla_select_table_where(cls, *args: BinaryExpression) -> list[Self]:
        if not args:
            where_clause = [True]
        else:
            where_clause = list(args)

        stmt = select(cls.sqla_table).where(*where_clause)

        # execute statement
        results = cls.sqla_execute(stmt)

        formatted_results = [
            cls(
                **{
                    column.name: getattr(obj[0], column.name)
                    for column in cls.sqla_table.__table__.columns
                }
            ) 
            for obj in results
        ]

        print(f"{len(formatted_results)} result{'s' if len(formatted_results) > 1 else ''} retrieved from '{cls.sqla_table.__tablename__}' table: {formatted_results}")
        return formatted_results
    
    # schema-level method to select from table with where clause
    @classmethod
    def sqla_select_where(cls, obj, *args: BinaryExpression) -> list[Self]:
        if not args:
            where_clause = [True]
        else:
            where_clause = list(args)

        stmt = select(obj).where(*where_clause)

        # execute statement
        results = cls.sqla_execute(stmt)

        return results
    
    # schema-level method to commit session with error catching
    @classmethod
    def _try_commit_session(cls, session: Session):
        try:
            # attempt to commit
            session.commit()
        except SQLAlchemyError as e:
            # Handle the general SQLAlchemy error
            session.rollback() # Important for maintaining session integrity
            print(f"An SQLAlchemy error occurred: {e}")
            raise

    # instance-level method to add instance as row to table
    def sqla_add(self):
        # construct session
        Session = sessionmaker(bind=self._engine)

        with Session() as session:
            # 4. Create an instance of your mapped class
            new_row = self.sqla_table(**self.model_dump())

            # 5. Add the instance to the session
            session.add(new_row)

            # 6. Commit the session to persist the changes to the database
            self._try_commit_session(session)

            print(f"Entry added to '{self.sqla_table.__tablename__}' table: {self.model_dump()}")

    # schema-level method to bulk add instances to a table
    @classmethod
    def sqla_add_all(cls, objs: list[Self]) -> list[Self]: 
        # construct session
        Session = sessionmaker(bind=cls._engine)

        with Session() as session:
            # map objects to table instances
            mapped_objs = [obj.sqla_table(**obj.model_dump()) for obj in objs]

            # add all objects to tables
            session.add_all(mapped_objs)

            # commit session to db
            cls._try_commit_session(session)

            print(f"{len(mapped_objs)} entries added to database: {mapped_objs}")

    # schema-level method to select from table
    @classmethod
    def sqla_delete_table_where(cls, *args: BinaryExpression) -> list[Self]:
        if not args:
            where_clause = [True]
        else:
            where_clause = list(args)

        # construct session
        Session = sessionmaker(bind=cls._engine)

        with Session() as session:
            objs_to_delete = session.query(cls.sqla_table).filter(*where_clause).all()
            
            for obj in objs_to_delete:
                session.delete(obj)

            # commit session to db
            cls._try_commit_session(session)

            print(f"{len(objs_to_delete)} entrie(s) deleted from '{cls.sqla_table.__tablename__}' table: {objs_to_delete}")
    
if __name__ == "__main__":
    pad_len = 150

    # SCHEMA DEFINITIONS -----------------------------------------------------------------
    # define application schema class with url
    class AppSchema(SqlaSchema):
        db_url = "sqlite:///test.db"
    AppSchema.construct_base_schema()
    # AppSchema = SqlaSchema.construct_base_schema(db_url="sqlite:///test.db")
    
    # define application class, User
    class User(AppSchema):
        id: Annotated[str, Field(description="user id"), Column(String)]
        time: Annotated[datetime, Field(description="Timestamp when user was added")]
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
        __table_args__ = (UniqueConstraint("id"),)
    Resource.sqla_create_table_class()
    
    # INIT DATABASE -----------------------------------------------------------------
    print(f"Initializing database ".ljust(pad_len, '-'))
    AppSchema.sqla_init_db()

    # ADD ENTRIES -----------------------------------------------------------------
    print(f"Adding entries ".ljust(pad_len, '-'))
    user_01 = User(id="test_user_01", time=datetime.now())
    user_02 = User(id="test_user_02", time=datetime.now())
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

    # test bulk add (can also be done at schema-level)
    AppSchema.sqla_add_all([user_02, resource_01, resource_02])
    # User.sqla_add_all([user_02, resource_01, resource_02])

    # SELECT TABLE ENTRIES -----------------------------------------------------------------
    print(f"Selecting table entries ".ljust(pad_len, '-'))
    # test select table
    users = User.sqla_select_table_where()
    print(users)
    
    resources = Resource.sqla_select_table_where()
    print(resources)
    
    # test select table with conditions
    resources = Resource.sqla_select_table_where((
        (Resource.sqla_table.id == 'test_resource_01') | 
        (Resource.sqla_table.enum_field == Color.BLUE)
    ))
    print(resources)
    
    # test select nested fields
    results = Resource.sqla_select_where(Resource.sqla_table.nested_model['id'], Resource.sqla_table.id == 'test_resource_01', Resource.sqla_table.enum_field == Color.BLUE)
    print(results)

    # DELETE ENTRIES -----------------------------------------------------------------
    print(f"Deleting entries ".ljust(pad_len, '-'))
    User.sqla_delete_table_where(User.sqla_table.id == 'test_user_02')
    print(User.sqla_select_table_where())

    # TRUNCATE TABLES -----------------------------------------------------------------
    print(f"Truncating tables ".ljust(pad_len, '-'))
    # test truncate on single table
    User.sqla_truncate_table()
    print(User.sqla_select_table_where())
    print(Resource.sqla_select_table_where())

    # test truncate on all tables
    AppSchema.sqla_truncate_table()
    print(User.sqla_select_table_where())
    print(Resource.sqla_select_table_where())