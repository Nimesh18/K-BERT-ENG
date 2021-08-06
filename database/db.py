import os
import re
from sqlalchemy import create_engine, text, event, exc
from sqlalchemy.pool import NullPool


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

    def __init__(self, connectionurl, predicate):
        # self.connectionurl = connectionurl not sure if the connection url would be required again after establishing the conneciton
        self.engine = create_engine(connectionurl, poolclass=NullPool)
        self.predicate = predicate

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


    def get_all_subjects_subset(self, limit=10):
        """
        return all subjects in db
        """
        self.engine.dispose()
        with self.engine.connect() as conn:
            res = conn.exec_driver_sql("SELECT DISTINCT predicate FROM temp_triples limit %(lim)s", dict(lim=limit))
            rows = res.all()
        return list(map(lambda x: x[0], rows))

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

