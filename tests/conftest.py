import ipaddress, socket
_original_getaddrinfo=socket.getaddrinfo
_original_connect=socket.socket.connect
def _local(host):
    if host in (None, '', 'localhost'):return True
    try:return ipaddress.ip_address(host).is_loopback
    except ValueError:return False
def _guard_lookup(host,*args,**kwargs):
    if not _local(host):raise RuntimeError('External network blocked in offline release tests')
    return _original_getaddrinfo(host,*args,**kwargs)
def _guard_connect(self,address):
    if isinstance(address,tuple) and not _local(address[0]):raise RuntimeError('External network blocked in offline release tests')
    return _original_connect(self,address)
socket.getaddrinfo=_guard_lookup
socket.socket.connect=_guard_connect
