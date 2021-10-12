import os
import re
from sqlalchemy import create_engine, text, event, exc
from sqlalchemy.pool import NullPool
from brain.config import CONCEPTNET, YAGO


# adding this in for now - ideally the values in the database should already be in this form: done at a later stage in the final database
def remove_quote(text):
    """
    removes beginning and ending quotations - "nationality"
    as well as language annotations, i.e @en
    """
    lang_filtered = re.sub("@en", "", text)
    quote_filtered = re.sub("^\"", "", lang_filtered)
    return re.sub("\"$", "", quote_filtered)

"""
Note!! The current database has predicate and subject erroneously swapped! So please change the relevant queries in future.
"""
class Database(object):

    def __init__(self, connectionurl, predicate, db_names):
        # self.connectionurl = connectionurl not sure if the connection url would be required again after establishing the conneciton
        self.engine = create_engine(connectionurl, poolclass=NullPool)
        self.predicate = predicate
        self.db_names = db_names

        @event.listens_for(self.engine, "connect")
        def connect(dbapi_connection, connection_record):
            connection_record.info['pid'] = os.getpid()

        @event.listens_for(self.engine, "checkout")
        def checkout(dbapi_connection, connection_record, connection_proxy):
            pid = os.getpid()
            if connection_record.info['pid'] != pid:
                connection_record.connection = connection_proxy.connection = None
                raise exc.DisconnectionError(
                        "Connection record belongs to pid %s, "
                        "attempting to check out in pid %s" %
                        (connection_record.info['pid'], pid)
                )

    def get_all_subjects(self):
        """
        return all subjects in db
        """
        self.engine.dispose()
        with self.engine.connect() as conn:
            res = conn.execute("SELECT DISTINCT predicate FROM temp_triples")
            rows = res.all()
        return list(map(lambda x: x[0], rows))


    # def get_all_subjects_subset(self, limit=10):
    #     """
    #     return all subjects in db
    #     """
    #     self.engine.dispose()
    #     with self.engine.connect() as conn:
    #         res = conn.exec_driver_sql("SELECT DISTINCT predicate FROM temp_triples limit %(lim)s", dict(lim=limit))
    #         rows = res.all()
    #     return list(map(lambda x: x[0], rows))

    def get_object(self, subject, limit):
        """
        get all predicate and object for a subject
        """
        self.engine.dispose()
        with self.engine.connect() as conn:
            res = conn.exec_driver_sql("SELECT object FROM temp_triples where predicate = %(subj)s limit %(lim)s", dict(subj=subject, lim=limit))
            rows = res.all()
        return list(map(lambda x: x[0], rows))

    def get(self, subject, limit):
        """
        get all predicate and object for a subject
        """
        self.engine.dispose()
        with self.engine.connect() as conn:
            res = conn.exec_driver_sql("SELECT subject, object FROM temp_triples where predicate = %(subj)s limit %(lim)s", dict(subj=subject, lim=limit))
            rows = res.all()
        return list(map(lambda x: remove_quote(x[0]) + " " + x[1], rows))

    def get_all_subjects_subset(self):
        """
        return all subjects in db
        """
        self.engine.dispose()
        with self.engine.connect() as conn:
            res = conn.exec_driver_sql("SELECT DISTINCT predicate FROM labels")
            rows = res.all()
        return list(map(lambda x: x[0], rows))

    def search(self, search_for, limit=None, offset=0):
        result_set = []
        for db_name in self.db_names:
            if db_name == CONCEPTNET:
                result_set.extend(self.search_conceptnet(search_for, limit, offset))
            elif db_name == YAGO:
                result_set.extend(self.search_yago(search_for, limit, offset))

        return result_set
    

    def search_yago(self, search_for, limit=None, offset=0):
        term = f"<http://yago-knowledge.org/resource/{search_for}%"
        self.engine.dispose()
        with self.engine.connect() as conn:
            query = "SELECT * FROM labels where predicate like %(term)s order by predicate, subject limit %(limit)s offset %(offset)s"
            res = conn.exec_driver_sql(query, dict(term=term, limit=limit, offset=offset))
            rows = res.all()
        converted_set = self.format_label_resultset(rows).values()
        return list(map(lambda x: ". ".join(x[::-1]), converted_set))

    # can remove once db swaps predicate and subject columns. Or just keep it if necessary
    def format_label_resultset(self, result_set):
        """
        result_set in form (id, predicate, subject, object)
        convert into dictionary, where predicate is key and unique objects are values
        """
        id, predicate, subject, obj = [0, 1, 2, 3]
        new_set = {}
        for result in result_set:
            if result[predicate] not in new_set:
                new_set[result[predicate]] = [result[obj]]
            else:
                new_set[result[predicate]].append(result[obj])
        return new_set

    def search_conceptnet(self, search_for, limit=None, offset=0):
        term = f"{search_for}"
        self.engine.dispose()
        with self.engine.connect() as conn:
            query = "SELECT * FROM triples where subject = %(term)s order by subject, predicate limit %(limit)s offset %(offset)s"
            res = conn.exec_driver_sql(query, dict(term=term, limit=limit, offset=offset))
            rows = res.all()
        # converted_set = self.format_label_resultsetv2(rows).values()
        return list(map(lambda x: " ".join(x), rows))

    def format_label_resultsetv2(self, result_set):
        """
        result_set in form (id, predicate, subject, object)
        convert into dictionary, where predicate is key and unique objects are values
        """
        id, subject, predicate, obj = [0, 1, 2, 3]
        new_set = {}
        for result in result_set:
            if result[subject] not in new_set:
                new_set[result[subject]] = [result[obj]]
            else:
                new_set[result[subject]].append(result[obj])
        return new_set

