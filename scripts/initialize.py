"""Validate the existing data package and idempotently initialize a separate DB."""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase, Query

from service.settings import Settings

EXPECTED_NODES = {'Movie': 156, 'Comment': 123693, 'Topic': 18, 'Sentiment': 3, 'Emotion': 7, 'User': 52773}
EXPECTED_RELS = {'REVIEWS': 123693, 'BELONGS_TO': 123693, 'HAS_SENTIMENT': 123693, 'HAS_EMOTION': 123693, 'POSTED': 123664}
NODE_SPECS = {'movies.csv': ('Movie', 'movie_id'), 'comments.csv': ('Comment', 'comment_id'), 'topics.csv': ('Topic','topic_id'), 'sentiments.csv': ('Sentiment','sentiment_id'), 'emotions.csv': ('Emotion','emotion_id'), 'users.csv': ('User','user_id')}
REL_SPECS = {
    'rel_comment_movie.csv': ('Comment','comment_id','Movie','movie_id','REVIEWS'),
    'rel_comment_topic.csv': ('Comment','comment_id','Topic','topic_id','BELONGS_TO'),
    'rel_comment_sentiment.csv': ('Comment','comment_id','Sentiment','sentiment_id','HAS_SENTIMENT'),
    'rel_comment_emotion.csv': ('Comment','comment_id','Emotion','emotion_id','HAS_EMOTION'),
    'rel_user_comment.csv': ('User','user_id','Comment','comment_id','POSTED'),
}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        yield from csv.DictReader(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_package(settings):
    started = time.monotonic()
    keys, counts, hashes = {}, {}, {}
    for filename, (label, key) in NODE_SPECS.items():
        path = settings.data_dir / filename
        ids = []
        for row in read_csv(path):
            identifier = row[key]
            if not identifier:
                raise ValueError(f'{filename}: 空主键')
            ids.append(identifier)
        if len(ids) != len(set(ids)):
            raise ValueError(f'{filename}: 重复主键')
        keys[label] = set(ids)
        counts[label] = len(ids)
        hashes[filename] = sha256(path)
    relations = {}
    for filename, (left, lk, right, rk, kind) in REL_SPECS.items():
        pairs = set()
        for row in read_csv(settings.data_dir / filename):
            pair = row[lk], row[rk]
            if pair in pairs or pair[0] not in keys[left] or pair[1] not in keys[right]:
                raise ValueError(f'{filename}: 重复关系或无效节点引用 {pair}')
            pairs.add(pair)
        relations[kind] = len(pairs)
        hashes[filename] = sha256(settings.data_dir / filename)
    if counts != EXPECTED_NODES or relations != EXPECTED_RELS:
        raise ValueError(f'数据规模与本版本不符: {counts}, {relations}')
    vectors = np.load(settings.embeddings_path, mmap_mode='r', allow_pickle=False)
    if vectors.shape != (counts['Comment'],384) or vectors.dtype != np.float32:
        raise ValueError(f'向量形状或类型不符: {vectors.shape}, {vectors.dtype}')
    for start in range(0, len(vectors), 4096):
        batch = vectors[start:start + 4096]
        if not np.isfinite(batch).all() or not np.allclose(np.linalg.norm(batch, axis=1), 1, atol=2e-5):
            raise ValueError(f'无效或未归一化向量，起始行 {start}')
    source = iter(read_csv(settings.preprocessed_path))
    for index, comment in enumerate(read_csv(settings.data_dir / 'comments.csv')):
        original = next(source, None)
        if original is None or comment['comment_id'] != f'c_{index}':
            raise ValueError(f'评论/向量行号不一致: {index}')
        for key in ('Comment','Movie_Name','Star','Date'):
            if comment[key] != original[key]:
                raise ValueError(f'预处理评论不一致: {index}, {key}')
    if next(source, None) is not None:
        raise ValueError('预处理数据有多余行')
    hashes['embeddings.npy'] = sha256(settings.embeddings_path)
    hashes['df_preprocessed.csv'] = sha256(settings.preprocessed_path)
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = {'fingerprint': fingerprint, 'nodes': counts, 'relationships': relations, 'node_total': sum(counts.values()), 'relationship_total': sum(relations.values()), 'vector_shape': list(vectors.shape), 'source_sha256': hashes, 'validation_seconds': round(time.monotonic()-started,3), 'validated_at': datetime.now().astimezone().isoformat(), 'userless_comments': 29}
    print(json.dumps({'validated': True, 'node_total': manifest['node_total'], 'relationship_total': manifest['relationship_total'], 'fingerprint': fingerprint}, ensure_ascii=False), flush=True)
    return manifest, vectors


def number(value, integer=False):
    if value is None or value == '' or value.lower() == 'nan':
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return int(value) if integer else value


def node_properties(label, row):
    if label == 'Comment':
        raw_date = row['Date'][:10]
        try:
            datetime.strptime(raw_date, '%Y-%m-%d')
        except ValueError:
            raw_date = None
        return {'comment_id': row['comment_id'], 'text': row['Comment'], 'star': number(row['Star'],True), 'like_count': number(row['Like'],True), 'comment_len': number(row['comment_len'],True), 'sentiment': row['sentiment'], 'emotion': row['bert_emotion_7class'], 'emotion_confidence': number(row['bert_confidence']), 'layer_consistency': row['layer_consistency'], 'topic_name': row['topic_name'], 'movie_name': row['Movie_Name'], 'date': raw_date}
    if label == 'Movie':
        return {k: (v if k == 'movie_id' else number(v, k == 'total_reviews')) for k,v in row.items() if k != 'movie_name'} | {'name': row['movie_name']}
    if label == 'Topic':
        return {'topic_id': row['topic_id'], 'name': row['topic_name'], 'keywords': row['keywords'], 'count': number(row['count'],True)}
    if label in ('Sentiment','Emotion'):
        prefix = label.lower()
        return {k:v for k,v in row.items() if k != prefix+'_name'} | {'name': row[prefix+'_name']}
    return {k: (v if k in ('user_id','username') else number(v, k in ('review_count','movie_count'))) for k,v in row.items()}


def batches(iterator, size=1500):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class Initializer:
    def __init__(self, settings):
        self.settings = settings
        self.driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user,settings.neo4j_password), max_transaction_retry_time=0, connection_timeout=10)

    def run(self, query, params=None):
        with self.driver.session(database=self.settings.neo4j_database) as session:
            result = session.run(Query(query, timeout=60), params or {})
            records = [record.data() for record in result]
            result.consume()
            return records

    def counts(self):
        nodes = {r['label']:r['n'] for r in self.run('MATCH (n) RETURN labels(n)[0] AS label,count(*) AS n')}
        rels = {r['kind']:r['n'] for r in self.run('MATCH ()-[r]->() RETURN type(r) AS kind,count(*) AS n')}
        vectors = self.run('MATCH (c:Comment) WHERE c.embedding IS NOT NULL RETURN count(c) AS n')[0]['n']
        return nodes, rels, vectors

    def import_nodes(self):
        for filename, (label,key) in NODE_SPECS.items():
            self.run(f'CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE')
            query = f'UNWIND $rows AS row MERGE (n:{label} {{{key}:row.{key}}}) SET n += row'
            count = 0
            for batch in batches(node_properties(label,row) for row in read_csv(self.settings.data_dir/filename)):
                self.run(query, {'rows':batch})
                count += len(batch)
            print(f'Imported {label}: {count}', flush=True)
        for property_name in ('movie_name','topic_name','sentiment','date'):
            self.run(f'CREATE INDEX comment_{property_name} IF NOT EXISTS FOR (c:Comment) ON (c.{property_name})')
        self.run('CREATE INDEX movie_name IF NOT EXISTS FOR (m:Movie) ON (m.name)')

    def import_relationships(self):
        for filename, (left,lk,right,rk,kind) in REL_SPECS.items():
            def transformed():
                for row in read_csv(self.settings.data_dir/filename):
                    properties = {}
                    if kind == 'BELONGS_TO':
                        properties['like_weight'] = number(row['like_weight'])
                    if kind == 'HAS_SENTIMENT':
                        properties['confidence'] = number(row['confidence'])
                    if kind == 'HAS_EMOTION':
                        properties = {'confidence': number(row['bert_confidence']), 'consistency': row['layer_consistency']}
                    yield {lk:row[lk], rk:row[rk], 'properties':properties}
            query = f'UNWIND $rows AS row MATCH (a:{left} {{{lk}:row.{lk}}}), (b:{right} {{{rk}:row.{rk}}}) MERGE (a)-[r:{kind}]->(b) SET r += row.properties'
            count = 0
            for batch in batches(transformed()):
                self.run(query, {'rows':batch})
                count += len(batch)
            print(f'Imported {kind}: {count}', flush=True)

    def create_vector_index(self):
        name = self.settings.vector_index
        if not name.replace('_','').isalnum():
            raise ValueError('无效索引名称')
        existing = self.run('SHOW VECTOR INDEXES YIELD name,options WHERE name=$name RETURN options', {'name':name})
        if existing and existing[0]['options']['indexConfig'].get('vector.quantization.enabled',False):
            # This is our independent DB's index, not the Desktop database.
            self.run(f'DROP INDEX {name} IF EXISTS')
        self.run(f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (c:Comment) ON (c.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`:384, `vector.similarity_function`:'cosine', `vector.quantization.enabled`:false}}}}")

    def wait_index(self, budget=3600):
        started = time.monotonic()
        while time.monotonic()-started < budget:
            index = self.run('SHOW VECTOR INDEXES YIELD name,state,populationPercent WHERE name=$name RETURN state,populationPercent', {'name':self.settings.vector_index})
            if index and index[0]['state'] == 'ONLINE':
                return index[0]
            if index and index[0]['state'] == 'FAILED':
                raise RuntimeError('原生向量索引 FAILED')
            time.sleep(2)
        raise TimeoutError('向量索引验证累计超过一小时；可启用 Chroma 备选')

    def import_vectors(self, vectors):
        self.create_vector_index()
        self.run('UNWIND $rows AS row MATCH (c:Comment {comment_id:row.id}) SET c.embedding=row.vector', {'rows':[{'id':f'c_{i}','vector':vectors[i].astype(float).tolist()} for i in range(64)]})
        self.wait_index()
        trial_start = time.monotonic()
        trial = self.run('CALL db.index.vector.queryNodes($index,5,$vector) YIELD node,score RETURN node.comment_id AS id,score', {'index':self.settings.vector_index,'vector':vectors[0].astype(float).tolist()})
        if not any(row['id']=='c_0' and row['score'] > .999 for row in trial):
            raise RuntimeError('原生向量索引小规模自检失败')
        print(f'Native vector trial passed in {time.monotonic()-trial_start:.2f}s', flush=True)
        for start in range(0,len(vectors),250):
            batch = [{'id':f'c_{i}','vector':vectors[i].astype(float).tolist()} for i in range(start,min(start+250,len(vectors)))]
            self.run('UNWIND $rows AS row MATCH (c:Comment {comment_id:row.id}) SET c.embedding=row.vector', {'rows':batch})
            if start % 10000 == 0:
                print(f'Imported vectors: {min(start+250,len(vectors))}/{len(vectors)}', flush=True)
        self.wait_index()

    def initialize(self, manifest, vectors):
        for attempt in range(60):
            try:
                self.driver.verify_connectivity()
                break
            except Exception:
                if attempt == 59:
                    raise
                time.sleep(2)
        state = self.run('MATCH (m:Movie) WHERE m._dataset_ready=true RETURN m._dataset_fingerprint AS fingerprint LIMIT 1')
        if state:
            if state[0]['fingerprint'] != manifest['fingerprint']:
                raise ValueError('已有数据库的数据包不同，请使用新的独立数据卷')
            nodes, rels, count = self.counts()
            if nodes != EXPECTED_NODES or rels != EXPECTED_RELS or count != len(vectors):
                raise ValueError('已完成的数据卷校验失败，停止而不覆盖数据')
            self.create_vector_index()
            self.wait_index()
            print('Already initialized; counts, relationships and vectors verified; no writes to corpus.', flush=True)
            return
        self.import_nodes()
        self.import_relationships()
        self.import_vectors(vectors)
        nodes,rels,count = self.counts()
        if nodes != EXPECTED_NODES or rels != EXPECTED_RELS or count != len(vectors):
            raise ValueError(f'导入后计数不符: {nodes}, {rels}, {count}')
        self.run('MATCH (m:Movie {movie_id:$id}) SET m._dataset_ready=true, m._dataset_fingerprint=$fingerprint, m._import_manifest=$manifest', {'id':'m_0','fingerprint':manifest['fingerprint'],'manifest':json.dumps(manifest,ensure_ascii=False)})
        print(f'Initialized: {sum(nodes.values())} nodes, {sum(rels.values())} relationships, {count} vectors', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--manifest',type=Path,default=Path('var/data-manifest.json'))
    args = parser.parse_args()
    settings = Settings()
    manifest,vectors = validate_package(settings)
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.validate_only:
        initializer = Initializer(settings)
        try:
            initializer.initialize(manifest,vectors)
        finally:
            initializer.driver.close()


if __name__ == '__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
