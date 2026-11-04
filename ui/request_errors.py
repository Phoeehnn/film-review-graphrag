"""User-facing failure descriptions without raw provider response bodies."""
import httpx


def failure_message(error):
    if isinstance(error,httpx.TimeoutException):
        return '连接超时。请求可能仍在完成，请稍后点击“重试刚才的问题”。'
    if isinstance(error,httpx.ConnectError):
        return '暂时连接不到服务。请确认项目容器已启动，再点击重试。'
    if isinstance(error,httpx.HTTPStatusError):
        status=error.response.status_code
        return {402:'模型账户余额不足，请补充余额后重试。',
            429:'请求较多，请稍等一会儿再重试。',
            409:'会话条件已更新，请刷新页面后重新提交问题。',
            404:'会话已过期，请重新提交问题以开始新会话。',
            422:'问题或筛选条件有冲突，请检查日期、星级范围和电影筛选后重新提交。',
            503:'问答服务暂未就绪。你仍可浏览已加载的数据；请稍后重试。'}.get(status,'服务暂时未能完成请求，请稍后重试。')
    code=str(error.get('code','')) if isinstance(error,dict) else ''
    if 'TIMEOUT' in code.upper():return '本次处理超时，已确认的依据仍保留。可以稍后重试或缩小问题范围。'
    if code=='SESSION_CONFLICT':return '会话条件已更新，请刷新页面后重新提交问题。'
    if code=='RateLimitError':return '模型服务请求较多，请稍后重试。'
    if code in ('AuthenticationError','PermissionDeniedError'):return '模型凭据不可用，请检查服务端配置后重试。'
    if code=='InsufficientBalanceError' or (isinstance(error,dict) and '402' in str(error.get('message',''))):
        return '模型账户余额不足，请补充余额后重试。'
    return '本次未能完成。请稍后重试；若持续失败，可缩小范围或换一种问法。'
