import ast
import json

import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import create_engine, text

from .. import ValidationError
from ..base import VannaBase
from ..types import TrainingPlan, TrainingPlanItem
from ..utils import deterministic_uuid


class PG_VectorStore(VannaBase):
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)

        config = config or {}

        if "connection_string" not in config:
            raise ValueError("A valid 'config' dictionary with a 'connection_string' is required.")

        self.connection_string = config.get("connection_string", None)
        if not self.connection_string:
            raise ValidationError("No connection string provided")
        self.n_results = config.get("n_results", 10)

        self.embedding_function = config.get("embedding_function")
        if not self.embedding_function:
            self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.table_schema = config.get("table_schema", "public")

        self.sql_collection = PGVector(
            embeddings=self.embedding_function,
            collection_name="sql",
            connection=self.connection_string,
            table_schema=self.table_schema,
        )

        self.ddl_collection = PGVector(
            embeddings=self.embedding_function,
            collection_name="ddl",
            connection=self.connection_string,
            table_schema=self.table_schema,
        )
        self.documentation_collection = PGVector(
            embeddings=self.embedding_function,
            collection_name="documentation",
            connection=self.connection_string,
            table_schema=self.table_schema,
        )

    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        question_sql_json = json.dumps({"question": question, "sql": sql}, ensure_ascii=False)
        _id = deterministic_uuid(question_sql_json) + "-sql"
        doc = Document(page_content=question_sql_json, metadata={"id": _id})
        self.sql_collection.add_documents([doc], ids=[doc.metadata["id"]])
        return _id

    def add_ddl(self, ddl: str, **kwargs) -> str:
        _id = deterministic_uuid(ddl) + "-ddl"
        doc = Document(page_content=ddl, metadata={"id": _id})
        self.ddl_collection.add_documents([doc], ids=[doc.metadata["id"]])
        return _id

    def add_documentation(self, documentation: str, **kwargs) -> str:
        _id = deterministic_uuid(documentation) + "-doc"
        doc = Document(page_content=documentation, metadata={"id": _id})
        self.documentation_collection.add_documents([doc], ids=[doc.metadata["id"]])
        return _id

    def get_collection(self, collection_name):
        match collection_name:
            case "sql":
                return self.sql_collection
            case "ddl":
                return self.ddl_collection
            case "documentation":
                return self.documentation_collection
            case _:
                raise ValueError("Specified collection does not exist.")

    def get_similar_question_sql(self, question: str) -> list:
        documents = self.sql_collection.similarity_search(query=question, k=self.n_results)
        return [ast.literal_eval(document.page_content) for document in documents]

    def get_related_ddl(self, question: str, **kwargs) -> list:
        documents = self.ddl_collection.similarity_search(query=question, k=self.n_results)
        return [document.page_content for document in documents]

    def get_related_documentation(self, question: str, **kwargs) -> list:
        documents = self.documentation_collection.similarity_search(query=question, k=self.n_results)
        return [document.page_content for document in documents]

    def train(
        self,
        question: str | None = None,
        sql: str | None = None,
        ddl: str | None = None,
        documentation: str | None = None,
        plan: TrainingPlan | None = None,
    ):
        
        if sql and question:
            print(f"Adding question: {question} and sql: {sql}")
            return self.add_question_sql(question=question, sql=sql)

        if documentation:
            print(f"Adding documentation: {documentation}")
            return self.add_documentation(documentation)

        if sql:
            question = self.generate_question(sql)
            print(f"Question generated with sql: {question}\nAdding SQL...")
            return self.add_question_sql(question=question, sql=sql)

        if ddl:
            print(f"Adding ddl: {ddl}")
            return self.add_ddl(ddl)

        if question:
            raise ValidationError("Please provide a SQL query.")

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL and item.item_name:
                    self.add_question_sql(question=item.item_name, sql=item.item_value)

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        # Establishing the connection
        engine = create_engine(self.connection_string)

        # Querying the 'langchain_pg_embedding' table with schema
        query_embedding = f"SELECT cmetadata, document FROM {self.table_schema}.langchain_pg_embedding"
        df_embedding = pd.read_sql(query_embedding, engine)

        # List to accumulate the processed rows
        processed_rows = []

        # Process each row in the DataFrame
        for _, row in df_embedding.iterrows():
            custom_id = row["cmetadata"]["id"]
            document = row["document"]
            training_data_type = "documentation" if custom_id[-3:] == "doc" else custom_id[-3:]

            if training_data_type == "sql":
                # Convert the document string to a dictionary
                try:
                    doc_dict = ast.literal_eval(document)
                    question = doc_dict.get("question")
                    content = doc_dict.get("sql")
                except (ValueError, SyntaxError):
                    print(f"Skipping row with custom_id {custom_id} due to parsing error.")
                    continue
            elif training_data_type in ["documentation", "ddl"]:
                question = None  # Default value for question
                content = document
            else:
                # If the suffix is not recognized, skip this row
                print(f"Skipping row with custom_id {custom_id} due to unrecognized training data type.")
                continue

            # Append the processed data to the list
            processed_rows.append({"id": custom_id, "question": question, "content": content, "training_data_type": training_data_type})

        # Create a DataFrame from the list of processed rows
        df_processed = pd.DataFrame(processed_rows)

        return df_processed

    def remove_training_data(self, id: str, **kwargs) -> bool:
        # Create the database engine
        engine = create_engine(self.connection_string)

        # SQL DELETE statement with schema
        delete_statement = text(
            f"""
            DELETE FROM {self.table_schema}.langchain_pg_embedding
            WHERE cmetadata ->> 'id' = :id
        """
        )

        # Connect to the database and execute the delete statement
        with engine.connect() as connection:
            # Start a transaction
            with connection.begin() as transaction:
                try:
                    result = connection.execute(delete_statement, {"id": id})
                    # Commit the transaction if the delete was successful
                    transaction.commit()
                    # Check if any row was deleted and return True or False accordingly
                    return result.rowcount > 0
                except Exception as e:
                    # Rollback the transaction in case of error
                    print(f"An error occurred: {e}")
                    transaction.rollback()
                    return False

    def remove_collection(self, collection_name: str) -> bool:
        engine = create_engine(self.connection_string)

        # Determine the suffix to look for based on the collection name
        suffix_map = {"ddl": "ddl", "sql": "sql", "documentation": "doc"}
        suffix = suffix_map.get(collection_name)

        if not suffix:
            print("Invalid collection name. Choose from 'ddl', 'sql', or 'documentation'.")
            return False

        # SQL query to delete rows based on the condition, with schema
        query = text(
            f"""
            DELETE FROM {self.table_schema}.langchain_pg_embedding
            WHERE cmetadata->>'id' LIKE '%{suffix}'
        """
        )

        # Execute the deletion within a transaction block
        with engine.connect() as connection:
            with connection.begin() as transaction:
                try:
                    result = connection.execute(query)
                    transaction.commit()  # Explicitly commit the transaction
                    if result.rowcount > 0:
                        print(f"Deleted {result.rowcount} rows from langchain_pg_embedding where collection is {collection_name}.")
                        return True
                    else:
                        print(f"No rows deleted for collection {collection_name}.")
                        return False
                except Exception as e:
                    print(f"An error occurred: {e}")
                    transaction.rollback()  # Rollback in case of error
                    return False

    def generate_embedding(self, *args, **kwargs):
        pass
