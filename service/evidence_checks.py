"""Bounded deterministic checks, not automatic Y/P/N grading.

Presence of a citation never establishes semantic relevance or factual correctness.
We can check IDs, explicit fractions, registered rates/ties and labelled metadata.
Unsupported interpretations still require a human reviewer.
"""
import json
import re

from .statistics import requires_statistics
from .topics import friendly_value, topic_display


EMOTIONS=('喜爱','感动','惊喜','中性','失望','愤怒','厌恶')
CITATION=r'\[((?:g|c)_\d+)\]'


def citation_check(answer,evidence):
    valid={item.id for item in evidence}
    cited=set(re.findall(CITATION,answer))
    issues=[]
    invalid=cited-valid
    if invalid:
        issues.append({'code':'INVALID_CITATION','message':'模型引用了未提供的证据：'+', '.join(sorted(invalid))})
        for identifier in invalid:
            answer=answer.replace(f'[{identifier}]','[引用无效]')
    if evidence and not (cited & valid):
        issues.append({'code':'MISSING_CITATION','message':'回答未引用本次证据'})
    return answer,issues


def confirmed_answer(evidence,reason=None):
    lines=['回答中有可核查字段与证据不一致，已改为以下证据摘要；综合解释需进一步复核。' if reason else '已确认的样本证据：']
    for item in evidence[:10]:
        data=item.data
        if item.kind=='comment':
            text=str(data.get('text','')).replace('\n',' ')[:220]
            lines.append(f"- {data.get('movie','未提供电影')}，{data.get('star','未知')}星，主题“{topic_display(data.get('topic',''))}”，预测情绪“{data.get('emotion','未知')}”：\n  > {text}\n\n  [{item.id}]")
        else:
            if data.get('definition'):
                lines.append(data['definition']+f' [{item.id}]')
            rates=data.get('rate_facts',[])
            if rates:
                for fact in rates[:12]:
                    lines.append(f"- {fact.get('movie') or '所选范围'}：{fact['label']} {fact['numerator']}/{fact['denominator']} = {fact['percent']}%。 [{item.id}]")
            else:
                rows=data.get('rows',[])
                for row in rows[:10]:
                    if row.get('tied_movie_count',0)>1:
                        lines.append(f"- 最高正面评论数为 {row['positive_count']}，共 {row['tied_movie_count']} 部电影并列；本次选“{row['movie']}”作代表，完整名单见统计证据。 [{item.id}]")
                    else:
                        visible={k:v for k,v in row.items() if not isinstance(v,dict) and k not in ('tied_movies',)}
                        lines.append('- '+json.dumps(friendly_value(visible),ensure_ascii=False)+f' [{item.id}]')
                if len(rows)>10:
                    lines.append(f'其余 {len(rows)-10} 行见完整统计证据。 [{item.id}]')
    if not evidence:
        lines.append('暂无已确认数据，请缩小问题范围后重试。')
    return '\n\n'.join(lines)


def validate_answer(question,strategy,answer,evidence):
    answer,issues=citation_check(answer,evidence)
    records={item.id:item for item in evidence}
    rates=[fact for item in evidence if item.kind=='aggregate' for fact in item.data.get('rate_facts',[])]
    # Citations conventionally follow punctuation. Associate that citation with
    # the preceding clause for validation while preserving the original answer.
    clauses=re.split(r'[。；\n]',re.sub(r'([。；])\s*(\[(?:g|c)_\d+\])',r' \2\1',answer))
    # Fractions in a statistical answer must use a registered numerator/denominator.
    for match in re.finditer(r'(?<![\d.])(\d+)\s*/\s*(\d+)(?![\d.])',answer):
        numerator,denominator=map(int,match.groups())
        nearby=answer[max(0,match.start()-45):match.end()+60]
        if not (rates or '%' in nearby or '％' in nearby):
            continue
        if not denominator or numerator>denominator:
            issues.append({'code':'INVALID_FRACTION','message':f'比例分子/分母不可核对：{numerator}/{denominator}'})
        elif rates and not any(numerator==f['numerator'] and denominator==f['denominator'] for f in rates):
            issues.append({'code':'WRONG_RATE_DENOMINATOR','message':f'比例 {numerator}/{denominator} 与本次统计口径不一致'})
        if denominator:
            values=[float(value) for value in re.findall(r'(\d+(?:\.\d+)?)\s*[%％]',nearby)]
            if values and not any(abs(value-100*numerator/denominator)<=.15 for value in values):
                issues.append({'code':'FRACTION_PERCENT_MISMATCH','message':f'比例 {numerator}/{denominator} 与邻近百分比不一致'})
    # Only percentages in sentences citing a statistic are checked. Comment text
    # may itself contain percentages; that is evidence text, not a computed rate.
    if rates:
        for sentence in clauses:
            identifiers=re.findall(CITATION,sentence)
            if not any(i in records and records[i].kind=='aggregate' for i in identifiers):
                continue
            for value in re.findall(r'(\d+(?:\.\d+)?)\s*[%％]',sentence):
                if not any(abs(float(value)-f['percent'])<=.15 for f in rates):
                    issues.append({'code':'UNSUPPORTED_PERCENT','message':f'统计回答中的 {value}% 不能从本次已注册比例核对'})
    for item in evidence:
        if item.kind=='aggregate':
            for row in item.data.get('rows',[]):
                ties=row.get('tied_movie_count',0)
                if ties>1 and ('并列' not in answer or str(ties) not in answer):
                    issues.append({'code':'TIES_OMITTED','message':f'最高数量有 {ties} 部并列，回答不能称唯一第一'})
    # Inspect explicit prediction claims, not subjective interpretations of text.
    for sentence in clauses:
        identifiers=re.findall(CITATION,sentence)
        prediction=re.search(r'(?:预测(?:情绪|标签)|BERT(?:判断|判定|情绪)|模型(?:标签|判为))\s*(?:为|是|[:：]|=|为)?\s*[“"*]*('+ '|'.join(EMOTIONS)+r')',sentence)
        for identifier in identifiers:
            item=records.get(identifier)
            if not item or item.kind!='comment':
                continue
            if prediction and item.data.get('emotion') and prediction.group(1)!=item.data['emotion']:
                issues.append({'code':'PREDICTION_LABEL_MISMATCH','message':f"[{identifier}] 的实际预测标签是{item.data['emotion']}，不能报告为{prediction.group(1)}"})
            star=re.search(r'(?:评分|星级)(?:为|是|[:：])?\s*([1-5])\s*星',sentence)
            if star and item.data.get('star') is not None and int(star.group(1))!=item.data['star']:
                issues.append({'code':'STAR_METADATA_MISMATCH','message':f'[{identifier}] 星级与元数据不一致'})
    # Strong claims use aggregate statistics; top-k comments cannot replace them.
    statistics_missing=requires_statistics(question) and not any(item.kind=='aggregate' and item.data.get('contains_statistics',True) for item in evidence)
    if strategy=='vector' and statistics_missing:
        issues.append({'code':'STATISTICS_UNAVAILABLE','message':'强制向量模式未执行统计查询，不能由召回评论计算全库数量、排名或比例'})
    # Stable de-duplication keeps traces short without hiding distinct checks.
    issues=list({(issue['code'],issue['message']):issue for issue in issues}.values())
    severe=any(i['code']!='STATISTICS_UNAVAILABLE' for i in issues)
    if severe:
        answer=confirmed_answer(evidence,reason=True)
    checks={'checked':['citation_ids','explicit_fraction_math','registered_percentages','registered_ties','explicit_prediction_labels','explicit_star_claims','statistics_availability'],
        'issues':issues,'corrected_answer':severe,'statistics_missing':statistics_missing,
        'limitation':'规则检查只覆盖明确字段；未自动证明评论语义相关、全部自然语言解释或自由Cypher业务口径正确，不是人工Y/P/N评分。'}
    return answer,[i['message'] for i in issues],checks
