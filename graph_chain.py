"""Compatibility entry points for the original thesis scripts. Start the API first."""
import httpx
from neo4j import GraphDatabase, Query, READ_ACCESS
from openai import OpenAI
from service.prompts import GRAPH_SCHEMA, FEW_SHOT_EXAMPLES
from service.safety import sanitize_cypher, ensure_read_summary
from service.settings import Settings

class Neo4jDB:
    def __init__(self):
        self.settings = Settings()
        self.driver = GraphDatabase.driver(self.settings.neo4j_uri,auth=(self.settings.neo4j_user,self.settings.neo4j_password),max_transaction_retry_time=0)
    def query(self,cypher,params=None):
        safe = sanitize_cypher(cypher)
        with self.driver.session(database=self.settings.neo4j_database,default_access_mode=READ_ACCESS) as session:
            ensure_read_summary(session.run(Query('EXPLAIN '+safe,timeout=10),params or {}).consume())
            return [record.data() for record in session.run(Query(safe,timeout=10),params or {})]
    def close(self):
        self.driver.close()

def get_llm_client():
    s = Settings()
    return OpenAI(api_key=s.deepseek_api_key,base_url=s.deepseek_base_url,max_retries=0,timeout=35)

def get_model_name():
    return Settings().deepseek_model

def text_to_cypher(question,client):
    response = client.chat.completions.create(model=get_model_name(),messages=[{'role':'system','content':GRAPH_SCHEMA+FEW_SHOT_EXAMPLES+'\n只输出只读 Cypher；禁止 CALL 和写操作。'},{'role':'user','content':question}],temperature=0,max_tokens=1400,extra_body={'thinking':{'type':'disabled'}})
    return (response.choices[0].message.content or '').strip()

def get_graph_response(question,chat_history=None):
    settings = Settings()
    with httpx.Client(timeout=settings.query_timeout+10) as client:
        response = client.post(settings.api_base_url.rstrip('/')+'/api/v1/query',json={'question':question,'strategy':'graph'})
        response.raise_for_status()
        result = response.json()
    rows = [row for e in result['evidence'] if e['kind']=='aggregate' for row in e['data'].get('rows',[])]
    return result['answer'],'\n'.join(result['cypher']),rows
