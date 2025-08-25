class Get :
    def __init__(self) :
        import psycopg2
        from Bulk_trainset_code.Utils.DBConfig import parameters
        self.CONN = psycopg2.connect(parameters.DBINFO)

    def TransferDF(self, query) :
        import psycopg2
        import pandas as pd

        try : df = pd.read_sql(query, con = self.CONN)
        except psycopg2.DatabaseError as ERROR: return print(ERROR)
        return df