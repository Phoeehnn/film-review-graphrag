import asyncio
import json
import time
from uuid import uuid4

from .engine import ANSWER_SYSTEM, measured, merge_subgraphs, redact_error
from .evidence_checks import confirmed_answer, validate_answer
from .llm import Usage
from .schemas import AgentRequest, QueryResult
from .tools import TOOL_DEFINITIONS, ToolRegistry
from .topics import topic_config, unsupported_scope_reason

AGENT_SYSTEM = ANSWER_SYSTEM + '''\n你可以调用六个检索工具。必须从工具结果确认事实，不能把对话历史当作数据库证据。
涉及主题先 list_topics；统计用图谱工具，评论解释用 search_comments。工具错误不是证据，按结构化错误修正参数。
每轮可以调用多个工具。最多五轮模型调用、十次工具执行；优先完成必要查询再回答。
工具结果中 evidence_ids 是本次可引用的统计证据 ID；search_comments 的 evidence_id 是真实评论 ID。
遇到工具能力无法提供的统计，明确说明不足，不要估计。'''


class AgentService:
    def __init__(self, engine, registry=None):
        self.engine = engine
        self.registry = registry or ToolRegistry(engine)

    async def ask(self, request: AgentRequest, started: float | None = None):
        started = started if started is not None else time.monotonic()
        usage = Usage()
        result = QueryResult(request_id=str(uuid4()),strategy='agent',status='ok')
        timings={}
        # All mutable conversation state is request-local; only clients are shared.
        messages = [{'role':'system','content':AGENT_SYSTEM+'\n主题显示映射：'+json.dumps(topic_config(),ensure_ascii=False)}]
        messages += [turn.model_dump() for turn in request.history[-12:]]
        messages.append({'role':'user','content':request.question})
        executions = 0
        exhausted = False
        try:
            async with asyncio.timeout(max(.001,self.engine.settings.agent_timeout-(time.monotonic()-started))):
                with measured(timings,'scope_validation'):
                    unsupported=unsupported_scope_reason(request.question)
                if unsupported:
                    result.status='cannot_answer'
                    result.answer=unsupported
                    result.warnings=['范围校验未通过，本次未调用模型或检索。']
                    result.evidence_checks={'checked':['supported_scope'],'issues':[{'code':'UNSUPPORTED_SCOPE','message':unsupported}],'corrected_answer':False}
                    result.elapsed_seconds=round(time.monotonic()-started,3)
                    result.stage_timings=timings
                    result.model_usage=usage.dump()
                    return result
                for round_no in range(1,self.engine.settings.max_agent_rounds+1):
                    with measured(timings,'agent_model'):
                        response = await self.engine.llm.complete(messages,usage,tools=TOOL_DEFINITIONS,tool_choice='auto')
                    message = response.choices[0].message
                    assistant = {'role':'assistant','content':message.content or ''}
                    if message.tool_calls:
                        assistant['tool_calls'] = [{'id':call.id,'type':'function','function':{'name':call.function.name,'arguments':call.function.arguments}} for call in message.tool_calls]
                    messages.append(assistant)
                    if not message.tool_calls:
                        with measured(timings,'evidence_validation'):
                            result.answer,validation,result.evidence_checks = validate_answer(request.question,'agent',message.content or '',result.evidence)
                        result.warnings += validation
                        if not result.evidence:
                            result.status = 'no_results' if result.tool_calls and all(t['ok'] and t['row_count']==0 for t in result.tool_calls) else 'insufficient_evidence'
                        elif validation:
                            result.status = 'partial'
                        break
                    for call in message.tool_calls:
                        if executions >= self.engine.settings.max_agent_tools:
                            exhausted = True
                            break
                        executions += 1
                        try:
                            arguments = json.loads(call.function.arguments)
                        except (ValueError,TypeError):
                            arguments = None
                        trace = {'round':round_no,'sequence':executions,'tool_call_id':call.id,'tool':call.function.name,'arguments':arguments}
                        begin = time.monotonic()
                        try:
                            outcome = await self.registry.execute(call.function.name,arguments,executions)
                        except Exception as exc:
                            outcome = {'ok':False,'error':{'code':type(exc).__name__,'message':redact_error(exc,self.engine.settings)},'evidence':[],'cypher':[],'subgraph':{'nodes':[],'relationships':[]}}
                        trace.update({'ok':outcome['ok'],'elapsed_seconds':round(time.monotonic()-begin,3),'row_count':len(outcome.get('rows',[])),'evidence_ids':[e.id for e in outcome['evidence']]})
                        timings['agent_tools']=round(timings.get('agent_tools',0)+time.monotonic()-begin,4)
                        if not outcome['ok']:
                            trace['error'] = outcome['error']
                        result.tool_calls.append(trace)
                        existing = {e.id for e in result.evidence}
                        result.evidence += [e for e in outcome['evidence'] if e.id not in existing]
                        result.cypher += outcome['cypher']
                        result.subgraph = merge_subgraphs(result.subgraph,outcome['subgraph'])
                        visible = {k:v for k,v in outcome.items() if k not in ('evidence','subgraph')}
                        messages.append({'role':'tool','tool_call_id':call.id,'content':json.dumps(visible,ensure_ascii=False)})
                    if exhausted:
                        break
                else:
                    exhausted = True
                if exhausted:
                    result.status = 'partial'
                    result.warnings.append('达到五轮模型或十次工具执行上限，未继续调用。')
                    result.answer = '已达到工具调用预算。以下是已确认的证据，综合解释尚未完成：\n'+self.confirmed(result)
        except TimeoutError:
            result.status = 'timeout'
            result.error = {'code':'AGENT_TIMEOUT','message':'Agent 超过 120 秒总时限'}
            result.answer = 'Agent 问答超时，已确认的证据保留：\n'+self.confirmed(result)
        except Exception as exc:
            result.status = 'error'
            result.error = {'code':type(exc).__name__,'message':redact_error(exc,self.engine.settings)}
            result.answer = 'Agent 运行失败，已确认的证据保留：\n'+self.confirmed(result)
        result.elapsed_seconds = round(time.monotonic()-started,3)
        result.stage_timings=timings
        result.model_usage = usage.dump()
        return result

    @staticmethod
    def confirmed(result):
        return confirmed_answer(result.evidence)
