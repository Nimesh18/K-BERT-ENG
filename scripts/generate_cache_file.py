import json
import csv
import argparse


ITEM_ID, LABELS, DESCRIPTIONS, ALIAS, SUBCLASS_VALUE, INSTANCE_VALUE, NAMED_ENT_ID, NAMED_ENT_VALUE = list(range(8))
NULL_STRING = '\\N'

def format_current_item_info(rows, term):
    """
    return list of entities [ent1, ent2]
    """
    if len(rows) == 0:
        return
    
    ents = [set([row[i] for row in rows]) for i in range(len(rows[0]))]
    unrolled = [entity for group in ents for entity in group]
    return list(filter(lambda x: x is not None and x != NULL_STRING and x != term, unrolled))


def format_current_group(rows, term):
    items = {}
    for row in rows:
        item_id = row[ITEM_ID]
        if item_id in items:
            items[item_id].append(row[LABELS:NAMED_ENT_ID])
        else:
            items[item_id] = [row[LABELS:NAMED_ENT_ID]]
    
    formatted_list = list(map(lambda item_info: format_current_item_info(item_info, term), items.values()))
    
    return list(filter(lambda x: len(x) > 0, formatted_list))

def format_current_group_selective(rows, term, props):
    """
    eg: props: [LABELS, DESCRIPTIONS, ALIAS, SUBCLASS_VALUE, INSTANCE_VALUE]
    exclude properties outside of props
    """
    items = {}
    for row in rows:
        item_id = row[ITEM_ID]
        if item_id in items:
            items[item_id].append(list(map(lambda x: row[x], props)))
        else:
            items[item_id] = [list(map(lambda x: row[x], props))]
    
    formatted_list = list(map(lambda item_info: format_current_item_info(item_info, term), items.values()))
    
    return list(filter(lambda x: len(x) > 0, formatted_list))

def create_cache(entityfile, props=None):
# counter = 20 'ag_news/train_cls_join.csv'
    current_term_id = None
    prev_term_id = None
    current_group = []
    cache = {}
    with open(entityfile, encoding='utf-8') as dd:
        reader = csv.reader(dd, delimiter='\t')
        # current_iter = 0
        for r in reader:
            current_term_id = r[NAMED_ENT_ID]
            current_term = r[NAMED_ENT_VALUE]
            if prev_term_id is not None and current_term_id != prev_term_id:
                if prev_term is not None and prev_term not in cache:
                    
                    cache[prev_term] = format_current_group(current_group, prev_term) if props is None else \
                            format_current_group_selective(current_group, prev_term, props)

                current_group = []
            
            current_group.append(r)
            prev_term_id = current_term_id
            prev_term = current_term
    return cache

def store_cache(cache, path):
    with open(path, 'w', encoding='utf-8') as ed:
        ed.write(json.dumps(cache))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_file', help=['path of output cache file'])
    parser.add_argument("--dataset", choices=["ag_news", "stsb"], default="ag_news",
                    help="Specify dataset" 
                            "AG NEWS SUBSET"
                            "STSB"
                            )

    parser.add_argument("--no_labels", action="store_true", help="Exclude labels", default=False)
    parser.add_argument("--no_alias", action="store_true", help="Exclude labels", default=False)
    parser.add_argument("--no_descriptions", action="store_true", help="Exclude labels", default=False)
    parser.add_argument("--no_subclassof", action="store_true", help="Exclude labels", default=False)
    parser.add_argument("--no_instanceof", action="store_true", help="Exclude labels", default=False)
    parser.add_argument('--table_join_file', help=['table join file'], default=None)

    args = parser.parse_args()
    if args.table_join_file is None:
        args.table_join_file = 'stsb/sts_table_join.csv' if args.dataset == "stsb" else 'ag_news/ag_table_join.csv'
    
    props = None
    opts = [args.no_labels, args.no_descriptions, args.no_alias, args.no_subclassof, args.no_instanceof]
    if any(opts):
        props = []
        for i, prop in enumerate([LABELS, DESCRIPTIONS, ALIAS, SUBCLASS_VALUE, INSTANCE_VALUE]):
            if not opts[i]:
                props.append(prop)

    stsb_cache = create_cache(args.table_join_file, props)
    store_cache(stsb_cache, args.output_file)


if __name__ == "__main__":
    main()