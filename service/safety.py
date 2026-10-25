"""Generated Cypher is untrusted. Routing READ is not an authorization boundary."""
import re


class UnsafeCypher(ValueError):
    pass


class ScopeUnsupported(UnsafeCypher):
    """A read-only query whose shape cannot carry the forced filters.

    It is still refused (subclass of UnsafeCypher), but it is not a write
    attempt; traces and reports count it separately from real unsafe output.
    """


def mask_literals(query: str) -> str:
    chars = list(query)
    i = 0
    while i < len(query):
        if query.startswith('//', i):
            end = query.find('\n', i)
            end = len(query) if end < 0 else end
        elif query.startswith('/*', i):
            end = query.find('*/', i + 2)
            if end < 0:
                raise UnsafeCypher('未闭合的注释')
            end += 2
        elif query[i] in ("'", '"', '`'):
            quote = query[i]
            end = i + 1
            while end < len(query):
                if query[end] == '\\' and quote != '`':
                    end += 2
                    continue
                if query[end] == quote:
                    if end + 1 < len(query) and query[end + 1] == quote:
                        end += 2
                        continue
                    end += 1
                    break
                end += 1
            else:
                raise UnsafeCypher('未闭合的字符串或标识符')
        else:
            i += 1
            continue
        for j in range(i, min(end, len(query))):
            if chars[j] != '\n':
                chars[j] = ' '
        i = end
    return ''.join(chars)


FORBIDDEN = re.compile(r'\b(?:DELETE|DETACH|CREATE|MERGE|SET|REMOVE|DROP|LOAD|FOREACH|CALL|SHOW|GRANT|DENY|REVOKE|ALTER|RENAME|USE|START|STOP|EXPLAIN|PROFILE)\b', re.I)


def sanitize_cypher(query: str, max_rows: int = 50) -> str:
    query = query.strip()
    if query.startswith('```'):
        query = re.sub(r'^```(?:cypher)?\s*', '', query, flags=re.I)
        query = re.sub(r'\s*```$', '', query).strip()
    if not query:
        raise UnsafeCypher('空查询')
    masked = mask_literals(query)
    semicolons = [m.start() for m in re.finditer(';', masked)]
    if semicolons:
        if len(semicolons) != 1 or masked[semicolons[0] + 1:].strip():
            raise UnsafeCypher('只允许一条查询')
        query = query[:semicolons[0]].rstrip()
        masked = mask_literals(query)
    if FORBIDDEN.search(masked):
        raise UnsafeCypher('查询包含写操作、过程调用或管理语句')
    if not re.match(r'^\s*(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN)\b', masked, re.I):
        raise UnsafeCypher('只允许只读检索语句')
    if not re.search(r'\bRETURN\b', masked, re.I):
        raise UnsafeCypher('查询必须返回结果')
    # Only alter top-level LIMIT. Nested limits can intentionally select a cohort.
    depth = 0
    top = []
    for i, char in enumerate(masked):
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
            if depth < 0:
                raise UnsafeCypher('括号不匹配')
        elif depth == 0:
            top.append((i, char))
    if depth:
        raise UnsafeCypher('括号不匹配')
    top_mask = list(' ' * len(masked))
    for i, char in top:
        top_mask[i] = char
    top_text = ''.join(top_mask)
    limits = list(re.finditer(r'\bLIMIT\b', top_text, re.I))
    has_union = bool(re.search(r'\bUNION\b', top_text, re.I))
    # Dynamic expressions/parameters for generated LIMIT cannot raise the bound.
    if limits:
        last = limits[-1]
        suffix = masked[last.end():].strip()
        literal = re.match(r'(\d+)\b', suffix)
        if not literal:
            raise UnsafeCypher('最外层 LIMIT 必须为整数常量')
        if suffix[literal.end():].strip():
            # This LIMIT truncates an intermediate cohort; cap the final rows.
            query += f'\nLIMIT {max_rows}'
        else:
            bound = min(int(literal.group(1)), max_rows)
            query = query[:last.end()] + f' {bound}'
    else:
        query += f'\nLIMIT {max_rows}'
    # A trusted wrapper also caps UNION output (LIMIT on its last branch is not global).
    if has_union:
        query = f'CALL {{\n{query}\n}}\nRETURN * LIMIT {max_rows}'
    return query


def ensure_read_summary(summary) -> None:
    if summary.query_type != 'r':
        raise UnsafeCypher(f'EXPLAIN 查询类型不是只读: {summary.query_type}')
