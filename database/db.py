import os
import json

ITEM_ID, LABELS, DESCRIPTIONS, ALIAS, SUBCLASS_VALUE, INSTANCE_VALUE, NAMED_ENT_ID, NAMED_ENT_VALUE = list(range(8))

class Database(object):

    def __init__(self, connectionurl, cache_path=None, cache_embedding_path=None):
        # self.connectionurl = connectionurl not sure if the connection url would be required again after establishing the conneciton
        self.cache_embeddings = {} if cache_embedding_path is None else self.load_cache(cache_embedding_path)

        if connectionurl is not None:
            from sqlalchemy import create_engine, text, event, exc
            from sqlalchemy.pool import NullPool
            self.engine = create_engine(connectionurl, poolclass=NullPool)
            self.cache = None

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

        else:
            self.cache = self.load_cache(cache_path)
            
    def format_item_info(self, rows, term):
        """
        return list of entities [ent1, ent2]
        """
        if len(rows) == 0:
            return []
        
        ents = [set([row[i] for row in rows]) for i in range(len(rows[0]))]
        unrolled = [entity for group in ents for entity in group]
        return list(filter(lambda x: x is not None and x != term, unrolled))

    def format_wiki_info(self, rows, term):
        items = {}
        for row in rows:
            item_id = row[0]
            if item_id in items:
                items[item_id].append(row[1:])
            else:
                items[item_id] = [row[1:]]

        formatted_list = list(map(lambda item_info: self.format_item_info(item_info, term), items.values()))
    
        return list(filter(lambda x: len(x) > 0, formatted_list))

    def load_cache(self, path):
        with open(path, encoding='utf-8') as fd:
            return json.loads(fd.readline())

    def retrieve_db_concepts(self, term):
        self.engine.dispose()
        with self.engine.connect() as conn:
            query = "SELECT i.id, i.labels, i.descriptions, a.value, s.value, ins.value FROM item i \
                LEFT JOIN alias a on i.id = a.item_id \
                LEFT JOIN item_instance ii on i.id = ii.item_id \
                LEFT JOIN instance ins on ii.instance_id = ins.code \
                LEFT JOIN item_subclass isu on i.id = isu.item_id \
                LEFT JOIN subclass s on isu.subclass_id = s.code \
                WHERE i.labels = %(term)s or a.value = %(term)s"
            res = conn.exec_driver_sql(query, dict(term=term))
            rows = res.all()
        entities = self.format_wiki_info(rows, term)
        return entities, self.cache_embeddings.get(term, [])

    def retrieve_wiki_concepts(self, search_for):
        term = f"{search_for.replace('_', ' ')}"
        
        if self.cache is not None:
            return self.cache.get(term, []), self.cache_embeddings.get(term, [])
            
        return self.retrieve_db_concepts(term)
        